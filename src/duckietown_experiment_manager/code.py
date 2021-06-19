import asyncio
import functools
import gc
import logging
import os
import shutil
import subprocess
import time
import traceback
from concurrent.futures.thread import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from tempfile import NamedTemporaryFile
from typing import cast, Dict, Iterator, List, Optional, Set

import cv2
import numpy as np
import stopit
import yaml
from zuper_commons.fs import locate_files, read_ustring_from_utf8_file, write_ustring_to_utf8_file
from zuper_commons.types import ZException, ZValueError
from zuper_ipce import IESO, ipce_from_object, object_from_ipce
from zuper_typing import can_be_used_as2

import duckietown_challenges as dc
from aido_analyze.utils_drawing import read_and_draw
from aido_analyze.utils_video import make_video2, make_video_ui_image
from aido_schemas import (
    DB20Observations,
    DB20ObservationsOnlyState,
    DB20ObservationsPlusState,
    DTSimStateDump,
    DumpState,
    EpisodeStart,
    GetCommands,
    GetDuckieState,
    GetRobotObservations,
    GetRobotState,
    JPGImage,
    protocol_agent_DB20,
    protocol_agent_DB20_fullstate,
    protocol_agent_DB20_onlystate,
    PROTOCOL_FULL,
    PROTOCOL_NORMAL,
    protocol_scenario_maker,
    protocol_simulator_DB20_timestamps,
    ProtocolDesc,
    RobotName,
    RobotObservations,
    RobotPerformance,
    RobotState,
    Scenario,
    SetMap,
    SetRobotCommands,
    SimulationState,
    SpawnDuckie,
    SpawnRobot,
    Step,
)
from aido_schemas.protocol_simulator import PROTOCOL_STATE
from aido_schemas.utils import TimeTracker
from duckietown_challenges import (
    ChallengeInterfaceEvaluator,
    ENV_CHALLENGE_NAME,
    ENV_SUBMISSION_ID,
    ENV_SUBMITTER_NAME,
    InvalidEnvironment,
    InvalidEvaluator,
    InvalidSubmission,
    json,
)
from duckietown_world.rules import EvaluatedMetric, RuleEvaluationResult
from zuper_nodes import RemoteNodeAborted
from zuper_nodes_wrapper.struct import MsgReceived
from zuper_nodes_wrapper.wrapper_outside import ComponentInterface
from . import logger
from .notice_thread import notice_thread
from .webserver import ImageWebServer

P = functools.partial


@dataclass
class MyConfig:
    episode_length_s: float
    min_episode_length_s: float
    seed: int
    physics_dt: float
    episodes_per_scenario: int
    max_failures: int

    fifo_dir: str

    sim_in: str
    sim_out: str
    sm_in: Optional[str]
    sm_out: Optional[str]

    timeout_initialization: int
    timeout_regular: int

    port: int  # port for visualization web server

    scenarios: List[str]

    do_webserver: bool = True


@contextmanager
def show_time(s: str):
    t0 = time.time()
    yield
    dt = time.time() - t0
    logger.debug(f"timer: {dt:.3f} for {s}")


def list_all_files(wd: str) -> List[str]:
    return [os.path.join(dp, f) for dp, dn, fn in os.walk(wd) for f in fn]


def check_existence_runner_file():
    MUST = "/fifos/runner"
    write_ustring_to_utf8_file("experiment_manager", "/fifos/experiment_manager")
    if not os.path.exists(MUST):
        msg = f"Path {MUST} does not exist"
        raise InvalidEnvironment(msg=msg, lf=list_all_files("/fifos"))


def set_loglevel():
    LEVELS = {
        "CRITICAL",
        "FATAL",
        "ERROR",
        "WARN",
        "WARNING",
        "INFO",
        "DEBUG",
        "NOTSET",
    }
    LOGLEVEL = os.environ.get("LOGLEVEL", "INFO").upper()
    if LOGLEVEL in LEVELS:
        # noinspection PyTypeChecker
        root_logger = logging.getLogger(None)
        root_logger.setLevel(LOGLEVEL)


async def main(cie: ChallengeInterfaceEvaluator, log_dir: str, attempts: str):
    set_loglevel()

    config_ = env_as_yaml("experiment_manager_parameters")
    config = cast(MyConfig, object_from_ipce(config_, MyConfig))
    logger.debug(config_yaml=config_, config_parsed=config)

    if "replica" in os.environ:
        replica = json.loads(os.environ["replica"])
        index = replica["index"]
        total = replica["total"]
    else:
        index = 0
        total = 1

    logger.debug(env=dict(os.environ), index=index, total=total)
    # check_existence_runner_file()

    if config.do_webserver:
        logger.debug("Running webserver")
        webserver = ImageWebServer(address="0.0.0.0", port=config.port)
        await asyncio.create_task(webserver.init())
    else:
        webserver = None

    if config.scenarios:
        logger.debug("using fixed scenarios")
        episodes0 = get_episodes_from_dirs(config.scenarios)
    else:
        logger.debug("using scenario maker")
        sm_ci = ComponentInterface(
            config.sm_in,
            config.sm_out,
            expect_protocol=protocol_scenario_maker,
            nickname="scenario_maker",
            timeout=config.timeout_regular,
        )

        # noinspection PyProtectedMember
        sm_ci._get_node_protocol(timeout=config.timeout_initialization)
        episodes0 = get_episodes(sm_ci, episodes_per_scenario=config.episodes_per_scenario, seed=config.seed)

    episodes = []
    for i, e in enumerate(episodes0):
        if i % total == index:
            logger.info(f"doing episode #{i} ({index}/{total})")
            episodes.append(e)
        else:
            logger.info(f"skipping episode #{i} ({index}/{total})")

    if not episodes:
        msg = "No episodes to do for me"
        logger.warn(msg)
        cie.set_score("skipped", 1)
        return
    all_player_robots: Set[RobotName] = set()
    all_controlled_robots: Dict[RobotName, str] = {}
    for episode_ in episodes:
        if not episode_.scenario.player_robots:
            raise ZValueError("no player robots in episode", episode=episode_)

        all_player_robots.update(episode_.scenario.player_robots)
        for name, r in episode_.scenario.robots.items():
            if r.controllable:
                if name in all_controlled_robots:
                    p0 = all_controlled_robots[name]
                    if p0 != r.protocol:
                        msg = f"Mismatch with protocols for robot {name}"
                        raise InvalidEvaluator(msg)
                else:
                    all_controlled_robots[name] = r.protocol
    # episode = episodes[0]

    msg = "Obtained episodes. Now initializing agents com."
    logger.debug(msg, all_player_robots=all_player_robots, all_controlled_robots=all_controlled_robots)
    agents_cis: Dict[str, ComponentInterface] = {}
    for robot_name, p in all_controlled_robots.items():
        fifo_in = os.path.join(config.fifo_dir, robot_name + "-in")
        fifo_out = os.path.join(config.fifo_dir, robot_name + "-out")

        if p == PROTOCOL_FULL:
            expect_protocol = protocol_agent_DB20_fullstate
        elif p == PROTOCOL_STATE:
            expect_protocol = protocol_agent_DB20_onlystate
        elif p == PROTOCOL_NORMAL:
            expect_protocol = protocol_agent_DB20
        else:
            raise NotImplementedError(p)
        # first open all fifos
        msg = f"Initializing agent"
        logger.debug(msg, robot_name=robot_name, fifo_in=fifo_in, fifo_out=fifo_out, protocol=p)
        with stopit.SignalTimeout(config.timeout_regular) as st:
            aci = ComponentInterface(
                fifo_in,
                fifo_out,
                expect_protocol=expect_protocol,
                nickname=robot_name,
                timeout=config.timeout_regular,
            )
        if not st:
            msg = f"Timeout during connection to {robot_name}: {st}"
            raise InvalidSubmission(msg)
        msg = f"Getting agent protocol"
        logger.debug(msg, robot_name=robot_name, fifo_in=fifo_in, fifo_out=fifo_out, protocol=p)

        try:
            # noinspection PyProtectedMember
            aci._get_node_protocol(timeout=config.timeout_initialization)
        except Exception as e:
            msg = f"Could not get protocol for agent {robot_name!r}"
            logger.error(msg)
            if robot_name in all_player_robots:
                raise InvalidSubmission(msg) from e
            elif robot_name in all_controlled_robots:
                raise InvalidEvaluator(msg) from e

        agents_cis[robot_name] = aci

    logger.debug("Now initializing sim connection", sim_in=config.sim_in, sim_out=config.sim_out)

    st: stopit.SignalTimeout
    with stopit.SignalTimeout(20) as st:
        sim_ci = ComponentInterface(
            config.sim_in,
            config.sim_out,
            expect_protocol=protocol_simulator_DB20_timestamps,
            nickname="simulator",
            timeout=config.timeout_regular,
        )
    if not st:
        msg = f"Timeout during connection: {st}"
        raise InvalidEvaluator(msg)
    try:
        logger.debug("Getting protocol for sim")

        try:
            # noinspection PyProtectedMember
            sim_ci._get_node_protocol(timeout=config.timeout_initialization)
        except Exception as e:
            msg = f"Cannot get protocol for simulator."
            logger.error(msg)
            raise InvalidEvaluator(msg) from e

        attempt_i: int = 0
        per_episode = {}
        stats = {}

        nfailures: int = 0
        logger.debug(f"Setting seed = {config.seed} for sim and agents.")
        sim_ci.write_topic_and_expect_zero("seed", config.seed)

        for pcname, robot_ci in agents_cis.items():
            try:
                robot_ci.write_topic_and_expect_zero("seed", config.seed)
            except RemoteNodeAborted as e:
                se = traceback.format_exc()

                if any(_ in se.lower() for _ in ["CUDA", "gpu", "out of memory"]):
                    msg = f"Detected out of CUDA memory:\n\n{se}"
                    raise InvalidEnvironment(msg) from e
                raise InvalidSubmission(msg) from e

        while episodes:

            if nfailures >= config.max_failures:
                msg = f"Too many failures: {nfailures}"
                raise InvalidEvaluator(msg)  # XXX

            episode_spec = episodes[0]
            episode_name = episode_spec.episode_name

            dn_final = os.path.join(log_dir, episode_name)

            if os.path.exists(dn_final):
                shutil.rmtree(dn_final)

            # dn = os.path.join(log_dir, episode_name + f".attempt{attempt_i}")
            dn = os.path.join(log_dir, episode_name)  #
            if os.path.exists(dn):
                shutil.rmtree(dn)

            if not os.path.exists(dn):
                os.makedirs(dn)
            fn = os.path.join(dn, "log.gs2.cbor")

            fn_tmp = fn + ".tmp"
            fw = open(fn_tmp, "wb")

            for pcname, agent_ci in agents_cis.items():
                agent_ci.cc(fw)
            sim_ci.cc(fw)

            logger.info(f"Now running episode {episode_name}")

            try:
                with show_time(f"run_episode"):
                    length_s = await run_episode(
                        sim_ci,
                        agents_cis,
                        episode_name=episode_name,
                        scenario=episode_spec.scenario,
                        config=config,
                        physics_dt=config.physics_dt,
                        webserver=webserver,
                    )
                logger.info(f"Finished episode {episode_name} with length {length_s:.2f}")

            except:
                msg = "Anomalous error from run_episode()"
                logger.error(msg, e=traceback.format_exc())
                raise
            finally:
                fw.close()
                os.rename(fn_tmp, fn)

            # output = os.path.join(dn, 'visualization')
            logger.debug("Now creating visualization and analyzing statistics.")

            if length_s > 0:
                with notice_thread("Make video", 2):
                    output_video = os.path.join(dn, "ui_image.mp4")
                    output_gif = os.path.join(dn, "ui_image.gif")
                    make_video_ui_image(log_filename=fn, output_video=output_video)
                    subprocess.check_call(["./makegif.sh", output_video, output_gif])

                # looks like if we load them before, procgraph does something funny
                with NamedTemporaryFile(suffix=".png") as f:
                    banner_bottom_fn = f.name

                    get_banner_bottom(banner_bottom_fn)
                    for pc_name in episode_spec.scenario.player_robots:
                        protocol = episode_spec.scenario.robots[pc_name].protocol

                        dn_i = os.path.join(dn, pc_name)
                        with notice_thread("Visualization", 2):
                            with show_time(f"visualization-{pc_name}"):
                                evaluated = read_and_draw(fn, dn_i, pc_name)

                        if len(evaluated) == 0:
                            msg = "Empty evaluated"
                            raise ZValueError(msg)

                        stats = {}
                        for k, evr in evaluated.items():
                            assert isinstance(evr, RuleEvaluationResult)
                            for m, em in evr.metrics.items():
                                assert isinstance(em, EvaluatedMetric)
                                assert isinstance(m, tuple)
                                if m:
                                    M = "/".join(m)
                                else:
                                    M = k
                                stats[M] = float(em.total)
                        per_episode[episode_name + "-" + pc_name] = stats
                        logger.debug(episode_name=episode_name, pc_name=pc_name, stats=stats)

                        if protocol in (PROTOCOL_FULL, PROTOCOL_NORMAL):
                            out_video = os.path.join(dn_i, "camera.mp4")
                            out_gif = os.path.join(dn_i, "camera.gif")
                            with show_time(f"video-{pc_name}"):
                                with notice_thread("Make video", 2):
                                    # make_video1(log_filename=fn, output_video=out_video, robot_name=pc_name)
                                    make_video2(
                                        log_filename=fn,
                                        output_video=out_video,
                                        robot_name=pc_name,
                                        banner_image="banner1.png",
                                        banner_image_bottom=banner_bottom_fn,
                                    )
                                    subprocess.check_call(["./makegif.sh", out_video, out_gif])

            if length_s >= config.min_episode_length_s:
                logger.info(f"{length_s:.1f} s are enough")
                episodes.pop(0)

                # logger.info(f"renaming {dn} -> {dn_final}")
                # os.rename(dn, dn_final)
            else:
                msg = f"episode too short with {length_s:.2f} s < {config.min_episode_length_s:.2f} s"
                logger.error(msg)

                nfailures += 1
            attempt_i += 1
    except dc.InvalidSubmission:
        raise
    except dc.InvalidEvaluator:
        raise
    except dc.InvalidConfiguration:
        raise
    except dc.InvalidEnvironment:
        raise
    except dc.AbortedByUser:
        raise
    except dc.AbortedByServer:
        raise
    except BaseException as e:
        msg = "Anomalous error while running episodes:"
        logger.error(msg, e=traceback.format_exc())
        raise dc.InvalidEvaluator(msg) from e

    finally:
        for agent_name, agent_ci in agents_cis.items():
            agent_ci.close()
        sim_ci.close()

    stats = list(stats)
    logger.info(per_episode=per_episode, stats=stats)
    if not per_episode or not stats:
        msg = "Could not find stats."
        logger.error(msg)
        raise InvalidEvaluator(msg, stats=stats, per_episode=per_episode)

    cie.set_score("per-episodes", per_episode)

    for k in list(stats):
        # logger.info(k=k, values=values)
        values = [_[k] for _ in per_episode.values() if k in _]

        cie.set_score(f"{k}_mean", float(np.mean(values)))
        cie.set_score(f"{k}_median", float(np.median(values)))
        cie.set_score(f"{k}_min", float(np.min(values)))
        cie.set_score(f"{k}_max", float(np.max(values)))


async def run_episode(
    sim_ci: ComponentInterface,
    agents_cis: Dict[RobotName, ComponentInterface],
    physics_dt: float,
    episode_name: str,
    scenario: Scenario,
    webserver: Optional[ImageWebServer],
    config: MyConfig,
) -> float:
    """returns length of episode"""
    logger.debug(scenario=scenario)
    episode_length_s = config.episode_length_s
    # clear simulation
    sim_ci.write_topic_and_expect_zero("clear")
    # set map data
    sim_ci.write_topic_and_expect_zero("set_map", SetMap(map_data=scenario.environment))

    controlled_robots: Dict[RobotName, ProtocolDesc] = {}  # protocol

    # spawn robot
    for robot_name, robot_conf in scenario.robots.items():
        if robot_conf.controllable:
            controlled_robots[robot_name] = robot_conf.protocol

        simulate_camera = robot_conf.protocol in (PROTOCOL_FULL, PROTOCOL_NORMAL)

        sp = SpawnRobot(
            robot_name=robot_name,
            configuration=robot_conf.configuration,
            playable=robot_conf.controllable,
            owned_by_player=robot_name in scenario.player_robots,
            color=robot_conf.color,
            simulate_camera=simulate_camera,
        )
        sim_ci.write_topic_and_expect_zero("spawn_robot", sp)
    for duckie_name, duckie_config in scenario.duckies.items():
        sp = SpawnDuckie(name=duckie_name, color=duckie_config.color, pose=duckie_config.pose)
        sim_ci.write_topic_and_expect_zero("spawn_duckie", sp)

    episode_start = EpisodeStart(episode_name, yaml_payload=scenario.payload_yaml)
    # start episode
    sim_ci.write_topic_and_expect_zero("episode_start", episode_start)

    for _, agent_ci in agents_cis.items():
        agent_ci.write_topic_and_expect_zero("episode_start", episode_start)

    current_sim_time: float = 0.0
    steps: int = 0
    # for now, fixed timesteps

    loop = asyncio.get_event_loop()

    stop_at = None
    with ThreadPoolExecutor(max_workers=5) as executor:
        while True:
            steps += 1
            if stop_at is not None:
                if steps == stop_at:
                    logger.info(f"Reached {steps} steps. Finishing. ")
                    break
            if current_sim_time >= episode_length_s:
                logger.info(f"Reached {episode_length_s:.1f} seconds. Finishing. ")
                break

            tt = TimeTracker(steps)
            t_effective = current_sim_time

            with tt.measure("complete-iteration"):

                with tt.measure("get_state_dump"):
                    f = P(sim_ci.write_topic_and_expect, "dump_state", DumpState(), expect="state_dump")
                    state_dump: MsgReceived[DTSimStateDump] = await loop.run_in_executor(executor, f)

                with tt.measure(f"get_robot_state"):
                    for robot_name in scenario.robots:
                        grs = GetRobotState(robot_name=robot_name, t_effective=t_effective)
                        f = P(sim_ci.write_topic_and_expect, "get_robot_state", grs, expect="robot_state")
                        _recv: MsgReceived[RobotState] = await loop.run_in_executor(executor, f)

                with tt.measure(f"get_duckie_state"):
                    for duckie_name in scenario.duckies:
                        rs = GetDuckieState(duckie_name, t_effective)
                        f = P(sim_ci.write_topic_and_expect, "get_duckie_state", rs, expect="duckie_state")
                        await loop.run_in_executor(executor, f)

                with tt.measure("sim_compute_sim_state"):
                    f = P(sim_ci.write_topic_and_expect, "get_sim_state", expect="sim_state")
                    recv: MsgReceived[SimulationState] = await loop.run_in_executor(executor, f)

                    sim_state: SimulationState = recv.data

                    if sim_state.done:
                        if stop_at is None:
                            NMORE = 15
                            stop_at = steps + NMORE
                            msg = (
                                f"Breaking because of simulator. Will break in {NMORE} more steps at step "
                                f"= {stop_at}."
                            )
                            logger.info(msg, sim_state=sim_state)
                        else:
                            msg = f"Simulation is done. Waiting for step {stop_at} to stop."
                            logger.info(msg)

                for agent_name in controlled_robots:
                    agent_ci = agents_cis[agent_name]
                    pr = controlled_robots[agent_name]
                    # for agent_name, agent_ci in agents_cis.items():

                    with tt.measure(f"sim_compute_performance-{agent_name}"):
                        f = P(
                            sim_ci.write_topic_and_expect,
                            "get_robot_performance",
                            agent_name,
                            expect="robot_performance",
                        )

                        _recv: MsgReceived[RobotPerformance] = await loop.run_in_executor(executor, f)

                    with tt.measure(f"sim_render-{agent_name}"):

                        if pr in (PROTOCOL_FULL, PROTOCOL_NORMAL):
                            get_robot_observations = GetRobotObservations(agent_name, t_effective)

                            f = P(
                                sim_ci.write_topic_and_expect,
                                "get_robot_observations",
                                get_robot_observations,
                                expect="robot_observations",
                            )
                            recv_observations: MsgReceived[RobotObservations]
                            recv_observations = await loop.run_in_executor(executor, f)
                            ro: RobotObservations = recv_observations.data
                            obs = cast(DB20Observations, ro.observations)
                            if webserver:
                                await webserver.push(f"{agent_name}-camera", obs.camera.jpg_data)
                        elif pr == PROTOCOL_STATE:
                            pass
                        else:
                            raise NotImplementedError(pr)

                    with tt.measure(f"agent_compute-{agent_name}"):
                        try:
                            map_data = cast(str, scenario.environment)

                            if pr == PROTOCOL_FULL:
                                obs_plus = DB20ObservationsPlusState(
                                    camera=obs.camera,
                                    odometry=obs.odometry,
                                    your_name=agent_name,
                                    state=state_dump.data.state,
                                    map_data=map_data,
                                )
                            elif pr == PROTOCOL_NORMAL:
                                obs_plus = DB20Observations(camera=obs.camera, odometry=obs.odometry)
                            elif pr == PROTOCOL_STATE:
                                obs_plus = DB20ObservationsOnlyState(
                                    your_name=agent_name,
                                    state=state_dump.data.state,
                                    map_data=map_data,
                                )
                            else:
                                raise NotImplementedError(pr)
                            # logger.info("sending agent", obs_plus=obs_plus)
                            agent_ci.write_topic_and_expect_zero("observations", obs_plus)
                            get_commands = GetCommands(t_effective)
                            f = P(
                                agent_ci.write_topic_and_expect,
                                "get_commands",
                                get_commands,
                                expect="commands",
                            )
                            with show_time(f"get_commands-{agent_name}"):
                                r: MsgReceived = await loop.run_in_executor(executor, f)
                        except BaseException as e:
                            msg = "Trouble with communication to the agent."
                            raise dc.InvalidSubmission(msg) from e

                    with tt.measure("set_robot_commands"):
                        set_robot_commands = SetRobotCommands(agent_name, t_effective, r.data)
                        f = P(
                            sim_ci.write_topic_and_expect_zero,
                            "set_robot_commands",
                            set_robot_commands,
                        )
                        with show_time(f"set_robot_commands-{agent_name}"):
                            await loop.run_in_executor(executor, f)

                with tt.measure("step_physics"):
                    current_sim_time += physics_dt
                    f = P(sim_ci.write_topic_and_expect_zero, "step", Step(current_sim_time))
                    with show_time(f"step_physics"):
                        await loop.run_in_executor(executor, f)

                with tt.measure("get_ui_image"):
                    f = P(sim_ci.write_topic_and_expect, "get_ui_image", None, expect="ui_image")
                    with show_time(f"get_ui_image"):
                        r_ui_image: MsgReceived[JPGImage] = await loop.run_in_executor(executor, f)

            if webserver:
                await webserver.push("ui_image", r_ui_image.data.jpg_data)
                await asyncio.sleep(0.05)
            log_timing_info(tt, sim_ci)

            gc.collect()

    return current_sim_time


def log_timing_info(tt, sim_ci: ComponentInterface):
    ieso = IESO(with_schema=False)
    ipce = ipce_from_object(tt, ieso=ieso)
    msg = {"compat": ["aido2"], "topic": "timing_information", "data": ipce}
    # noinspection PyProtectedMember
    j = sim_ci._serialize(msg)
    # noinspection PyProtectedMember
    sim_ci._cc.write(j)
    # noinspection PyProtectedMember
    sim_ci._cc.flush()


def check_compatibility_between_agent_and_sim(agent_ci: ComponentInterface, sim_ci: ComponentInterface):
    snp = sim_ci.node_protocol
    type_observations_sim = snp.outputs["robot_observations"].__annotations__["observations"]
    type_commands_sim = snp.inputs["set_robot_commands"].__annotations__["commands"]

    logger.debug(f"Simulation provides observations {type_observations_sim}")
    logger.debug(f"Simulation requires commands {type_commands_sim}")

    if agent_ci.node_protocol is None:
        msg = "Cannot check compatibility of interfaces because the agent does not implement reflection."
        logger.warning(msg)
        agent_ci.expect_protocol.outputs["commands"] = type_commands_sim
        agent_ci.expect_protocol.inputs["observations"] = type_observations_sim
        return

    type_observations_agent = agent_ci.node_protocol.inputs["observations"]
    # logger.info(f"Agent requires observations", type_observations_agent=type_observations_agent)

    type_commands_agent = agent_ci.node_protocol.outputs["commands"]
    # logger.info(f"Agent provides commands", type_commands_agent=type_commands_agent)

    r = can_be_used_as2(type_observations_sim, type_observations_agent)
    if not r.result:
        msg = "Observations mismatch"
        logger.error(msg, r=r, agent_protocol=agent_ci.node_protocol)
        raise ZException(msg, r=r)
    r = can_be_used_as2(type_commands_agent, type_commands_sim)
    if not r:
        msg = "Commands mismatch"
        logger.error(msg, r=r, agent_protocol=agent_ci.node_protocol)
        raise ZException(msg, r=r)


@dataclass
class EpisodeSpec:
    episode_name: str
    scenario: Scenario


def get_episodes_from_dirs(dirs: List[str]) -> List[EpisodeSpec]:
    pattern = "*scenario.yaml"
    episodes = []
    for d in dirs:
        if not os.path.exists:
            raise ZValueError("dir does not exist", d=d)
        filenames = locate_files(d, pattern)
        if not filenames:
            msg = "No files found in directory"
            raise ZValueError(msg, d=d, pattern=pattern, all_files=locate_files(d, "*.*"))
        for f in filenames:
            dn = os.path.dirname(f)
            episode_name = os.path.basename(dn)
            config_ = yaml.load(read_ustring_from_utf8_file(f), Loader=yaml.Loader)
            scenario = cast(Scenario, object_from_ipce(config_, Scenario))
            episode_spec = EpisodeSpec(episode_name, scenario)
            episodes.append(episode_spec)

    return episodes


def get_episodes(sm_ci: ComponentInterface, episodes_per_scenario: int, seed: int) -> List[EpisodeSpec]:
    sm_ci.write_topic_and_expect_zero("seed", seed)

    def iterate_scenarios() -> Iterator[Scenario]:
        while True:
            recv = sm_ci.write_topic_and_expect("next_scenario")
            if recv.topic == "finished":
                sm_ci.close()
                break
            else:
                yield recv.data

    episodes = []
    for scenario in iterate_scenarios():
        scenario_name = scenario.scenario_name
        # logger.info(f"Received scenario {scenario}")
        for i in range(episodes_per_scenario):
            episode_name = f"{scenario_name}-{i}"
            es = EpisodeSpec(episode_name=episode_name, scenario=scenario)
            episodes.append(es)
    return episodes


def env_as_yaml(name: str) -> dict:
    environment = os.environ.copy()
    if not name in environment:
        msg = f'Could not find variable "{name}".'
        raise ZValueError(msg, environment=dict(environment))
    v = environment[name]
    try:
        return yaml.load(v, Loader=yaml.SafeLoader)
    except Exception as e:
        msg = f"Could not load YAML."
        raise ZException(msg, v=v) from e


def get_banner_bottom(banner_bottom_fn: str):
    from procgraph_pil import imread, imwrite

    banner_bottom_template = "banner_bottom_template.png"
    rgb = imread(banner_bottom_template)

    font = cv2.FONT_HERSHEY_SIMPLEX
    submitter_name = os.environ[ENV_SUBMITTER_NAME]
    submission_id = os.environ[ENV_SUBMISSION_ID]
    challenge_name = os.environ[ENV_CHALLENGE_NAME]
    s = f"{submitter_name}\nchallenge {challenge_name}\nsubmission {submission_id}"

    color = (0, 0, 0)
    draw_text(
        rgb,
        text=s,
        uv_top_left=(10, 10),
        color=color,
        outline_color=None,
        fontFace=font,
        fontScale=0.8,
        line_spacing=1.7,
        thickness=1,
    )

    # cv2.putText(rgb, f"submission {submission_id}", (5, H), font, font_scale, (0, 0, 0), 1, cv2.LINE_AA)

    imwrite(rgb, banner_bottom_fn)


def draw_text(
    img,
    *,
    text,
    uv_top_left,
    color=(255, 255, 255),
    fontScale=0.5,
    thickness=1,
    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
    outline_color=(0, 0, 0),
    line_spacing=1.5,
):
    """
    Draws multiline with an outline.
    """
    assert isinstance(text, str)

    uv_top_left = np.array(uv_top_left, dtype=float)
    assert uv_top_left.shape == (2,)

    for line in text.splitlines():
        (w, h), _ = cv2.getTextSize(
            text=line,
            fontFace=fontFace,
            fontScale=fontScale,
            thickness=thickness,
        )
        uv_bottom_left_i = uv_top_left + [0, h]
        org = tuple(uv_bottom_left_i.astype(int))

        if outline_color is not None:
            cv2.putText(
                img,
                text=line,
                org=org,
                fontFace=fontFace,
                fontScale=fontScale,
                color=outline_color,
                thickness=thickness * 3,
                lineType=cv2.LINE_AA,
            )
        cv2.putText(
            img,
            text=line,
            org=org,
            fontFace=fontFace,
            fontScale=fontScale,
            color=color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )

        uv_top_left += [0, h * line_spacing]
