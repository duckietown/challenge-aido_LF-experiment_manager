#!/usr/bin/env python

import asyncio
import functools
import json
import os
import shutil
import time
import traceback
from concurrent.futures.thread import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Thread
from typing import Callable, cast, Dict, Iterator, List

import numpy as np
import yaml
from zuper_commons.logs import ZLogger
from zuper_commons.types import ZException
from zuper_ipce import ipce_from_object, object_from_ipce
from zuper_nodes.structures import RemoteNodeAborted
from zuper_nodes_wrapper.struct import MsgReceived
from zuper_nodes_wrapper.wrapper_outside import ComponentInterface
from zuper_typing import can_be_used_as2

from aido_analyze.utils_drawing import read_and_draw
from aido_analyze.utils_video import make_video1, make_video_ui_image
from aido_schemas import (
    DB20Observations,
    DB20ObservationsPlusState,
    DTSimStateDump,
    EpisodeStart,
    GetCommands,
    GetRobotObservations,
    GetRobotState,
    JPGImage,
    protocol_agent_DB20,
    protocol_scenario_maker,
    protocol_simulator_DB20,
    RobotObservations,
    RobotPerformance,
    RobotState,
    Scenario,
    SetMap,
    SetRobotCommands,
    SimulationState,
    SpawnRobot,
    Step,
)
from aido_schemas.protocol_simulator import DumpState
from aido_schemas.utils import TimeTracker
from duckietown_challenges import ChallengeInterfaceEvaluator
from duckietown_world import construct_map, draw_static, DuckietownMap
from duckietown_world.rules import RuleEvaluationResult
from duckietown_world.rules.rule import EvaluatedMetric
from webserver import WebServer

logger = ZLogger("experiment_manager")
__version__ = "6.0.17"

logger.info(f"experiment_manager {__version__}")


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
    sm_in: str
    sm_out: str

    timeout_initialization: int
    timeout_regular: int

    port: int  # port for visualization web server


async def main(cie: ChallengeInterfaceEvaluator, log_dir: str, attempts: str):
    config_ = env_as_yaml("experiment_manager_parameters")

    config = cast(MyConfig, object_from_ipce(config_, MyConfig))
    logger.info(config_yaml=config_, config_parsed=config)
    webserver = WebServer(address="0.0.0.0", port=config.port)
    await asyncio.create_task(webserver.init())

    sm_ci = ComponentInterface(
        config.sm_in,
        config.sm_out,
        expect_protocol=protocol_scenario_maker,
        nickname="scenario_maker",
        timeout=config.timeout_regular,
    )
    # noinspection PyProtectedMember
    sm_ci._get_node_protocol(timeout=config.timeout_initialization)
    episodes = get_episodes(sm_ci, episodes_per_scenario=config.episodes_per_scenario, seed=config.seed)

    if False:
        for ep in episodes:
            yaml_str = ep.scenario.environment
            map_data = yaml.load(yaml_str, Loader=yaml.SafeLoader)
            dm: DuckietownMap = construct_map(map_data)
            draw_maps_dir = os.path.join(log_dir, "draw_maps")
            for robot, rp in ep.scenario.robots.items():
                pose = rp.configuration.pose
            #
            # robot = DB18(color=color)
            # # noinspection PyTypeChecker
            # duckietown_map.set_object(robot_name, robot, ground_truth=trajs.pose)

            fns = draw_static(dm, output_dir=draw_maps_dir, pixel_size=(640, 640), area=None)
            for fn in fns:
                logger.info(f"Written to {fn}")
            pass

    episode = episodes[0]
    playable_robots = []
    for name, r in episode.scenario.robots.items():
        if r.playable:
            playable_robots.append(name)

    logger.info(
        "Obtained episodes from scenario maker. Now initializing agents com.", playable_robots=playable_robots
    )
    agents_cis: Dict[str, ComponentInterface] = {}
    for robot_name in playable_robots:
        fifo_in = os.path.join(config.fifo_dir, robot_name + "-in")
        fifo_out = os.path.join(config.fifo_dir, robot_name + "-out")

        # first open all fifos
        logger.info(f"Initializing agent", robot_name=robot_name, fifo_in=fifo_in, fifo_out=fifo_out)
        aci = ComponentInterface(
            fifo_in,
            fifo_out,
            expect_protocol=protocol_agent_DB20,
            nickname=robot_name,
            timeout=config.timeout_regular,
        )
        agents_cis[robot_name] = aci

    logger.info("Now initializing sim connection", sim_in=config.sim_in, sim_out=config.sim_out)

    sim_ci = ComponentInterface(
        config.sim_in,
        config.sim_out,
        expect_protocol=protocol_simulator_DB20,
        nickname="simulator",
        timeout=config.timeout_regular,
    )

    logger.info("Getting protocol for sim")
    # noinspection PyProtectedMember
    sim_ci._get_node_protocol(timeout=config.timeout_initialization)

    for pcname, robot_ci in agents_cis.items():
        logger.info("Getting protocol for agent")
        # noinspection PyProtectedMember
        robot_ci._get_node_protocol(timeout=config.timeout_initialization)
        if True:
            check_compatibility_between_agent_and_sim(robot_ci, sim_ci)

    attempt_i = 0
    per_episode = {}
    stats = {}
    try:

        nfailures = 0
        logger.info(f"Setting seed = {config.seed} for sim")
        sim_ci.write_topic_and_expect_zero("seed", config.seed)

        for pcname, robot_ci in agents_cis.items():
            logger.info(f"Setting seed = {config.seed} for agent")
            robot_ci.write_topic_and_expect_zero("seed", config.seed)

        while episodes:

            if nfailures >= config.max_failures:
                msg = f"Too many failures: {nfailures}"
                raise Exception(msg)  # XXX

            episode_spec = episodes[0]
            episode_name = episode_spec.episode_name

            logger.info(f"Starting episode {episode_name}")

            dn_final = os.path.join(log_dir, episode_name)

            if os.path.exists(dn_final):
                shutil.rmtree(dn_final)

            dn = os.path.join(attempts, episode_name + f".attempt{attempt_i}")
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

            logger.info("Now running episode")

            num_playable = len([_ for _ in episode_spec.scenario.robots.values() if _.playable])
            if num_playable != len(playable_robots):
                msg = (
                    f"The scenario requires {num_playable} robots,"
                    f"but I only know "
                    f"{len(playable_robots)} agents"
                )
                raise Exception(msg)  # XXX
            try:
                logger.info(f"Starting episode {episode_name}")
                length_s = await run_episode(
                    sim_ci,
                    agents_cis,
                    episode_name=episode_name,
                    scenario=episode_spec.scenario,
                    config=config,
                    physics_dt=config.physics_dt,
                    webserver=webserver,
                )
                logger.info(f"Finished episode {episode_name} with length {length_s}")

            except:
                msg = "Anomalous error from run_episode()"
                logger.error(msg, e=traceback.format_exc())
                raise
            finally:
                fw.close()
                os.rename(fn_tmp, fn)

            # output = os.path.join(dn, 'visualization')
            logger.info("Now creating visualization and analyzing statistics.")

            if length_s > 0:
                with notice_thread("Make video", 2):
                    make_video_ui_image(log_filename=fn, output_video=os.path.join(dn, "ui_image.mp4"))

                for pc_name in playable_robots:

                    dn_i = os.path.join(dn, pc_name)
                    with notice_thread("Visualization", 2):
                        evaluated = read_and_draw(fn, dn_i, pc_name)

                    out_video = os.path.join(dn_i, "camera.mp4")
                    with notice_thread("Make video", 2):
                        make_video1(log_filename=fn, output_video=out_video, robot_name=pc_name)

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
                    logger.info(episode_name=episode_name, pc_name=pc_name, stats=stats)

            if length_s >= config.min_episode_length_s:
                logger.info(f"{length_s:1f} s are enough")
                episodes.pop(0)

                os.rename(dn, dn_final)
            else:
                msg = f"episode too short with {length_s:1f} s < {config.min_episode_length_s:.1f} s"
                logger.error(msg)

                nfailures += 1
            attempt_i += 1
    except dc.InvalidSubmission:
        raise
    except BaseException as e:
        msg = "Anomalous error while running episodes:"
        logger.error(msg, e=traceback.format_exc())
        raise dc.InvalidEvaluator(msg) from e

    finally:
        for agent_name, agent_ci in agents_cis.items():
            agent_ci.close()
        sim_ci.close()
        logger.info("Simulation done.")

    logger.info(per_episode=per_episode, stats=list(stats))
    cie.set_score("per-episodes", per_episode)

    for k in list(stats):
        # logger.info(k=k, values=values)
        values = [_[k] for _ in per_episode.values()]
        cie.set_score(f"{k}_mean", float(np.mean(values)))
        cie.set_score(f"{k}_median", float(np.median(values)))
        cie.set_score(f"{k}_min", float(np.min(values)))
        cie.set_score(f"{k}_max", float(np.max(values)))


@contextmanager
def notice_thread(msg, interval):
    stop = False
    t0 = time.time()
    t = Thread(target=notice_thread_child, args=(msg, interval, lambda: stop))
    t.start()
    try:

        yield

    finally:
        t1 = time.time()
        delta = t1 - t0
        logger.info(f"{msg}: took {delta} seconds.")
        stop = True
        logger.info("waiting for thread to finish")
        t.join()


def notice_thread_child(msg: str, interval: float, stop_condition: Callable[[], bool]) -> None:
    t0 = time.time()
    while not stop_condition():
        delta = time.time() - t0
        logger.info(msg + f"(running for {int(delta)} seconds)")
        time.sleep(interval)
    # logger.info('notice_thread_child finishes')


async def run_episode(
    sim_ci: ComponentInterface,
    agents_cis: Dict[str, ComponentInterface],
    physics_dt: float,
    episode_name: str,
    scenario: Scenario,
    webserver: WebServer,
    config: MyConfig,
) -> float:
    """ returns length of episode """
    episode_length_s = config.episode_length_s
    # clear simulation
    sim_ci.write_topic_and_expect_zero("clear")
    # set map data
    sim_ci.write_topic_and_expect_zero("set_map", SetMap(map_data=scenario.environment))

    # spawn robot
    for robot_name, robot_conf in scenario.robots.items():
        sim_ci.write_topic_and_expect_zero(
            "spawn_robot",
            SpawnRobot(
                robot_name=robot_name,
                configuration=robot_conf.configuration,
                playable=robot_conf.playable,
                motion=robot_conf.motion,
            ),
        )

    # start episode
    sim_ci.write_topic_and_expect_zero("episode_start", EpisodeStart(episode_name))

    for _, agent_ci in agents_cis.items():
        agent_ci.write_topic_and_expect_zero("episode_start", EpisodeStart(episode_name))

    current_sim_time = 0.0

    # for now, fixed timesteps

    steps = 0

    loop = asyncio.get_event_loop()

    not_playable_robots = [_ for _ in scenario.robots if not scenario.robots[_].playable]
    with ThreadPoolExecutor(max_workers=5) as executor:
        while True:
            if current_sim_time >= episode_length_s:
                logger.info(f"Reached {episode_length_s:1f} seconds. Finishing. ")
                break

            tt = TimeTracker(steps)
            t_effective = current_sim_time

            with tt.measure("complete-iteration"):

                with tt.measure("get_state_dump"):
                    f = functools.partial(
                        sim_ci.write_topic_and_expect, "dump_state", DumpState(), expect="state_dump"
                    )
                    state_dump: MsgReceived[DTSimStateDump] = await loop.run_in_executor(executor, f)

                for agent_name, agent_ci in agents_cis.items():

                    # have this first, so we have something for t = 0
                    with tt.measure(f"sim_compute_robot_state-{agent_name}"):

                        grs = GetRobotState(robot_name=agent_name, t_effective=t_effective)
                        f = functools.partial(
                            sim_ci.write_topic_and_expect, "get_robot_state", grs, expect="robot_state",
                        )

                        _recv: MsgReceived[RobotState] = await loop.run_in_executor(executor, f)

                    with tt.measure(f"sim_compute_performance-{agent_name}"):
                        f = functools.partial(
                            sim_ci.write_topic_and_expect,
                            "get_robot_performance",
                            agent_name,
                            expect="robot_performance",
                        )

                        _recv: MsgReceived[RobotPerformance] = await loop.run_in_executor(executor, f)

                    with tt.measure(f"sim_render-{agent_name}"):
                        get_robot_observations = GetRobotObservations(agent_name, t_effective)

                        f = functools.partial(
                            sim_ci.write_topic_and_expect,
                            "get_robot_observations",
                            get_robot_observations,
                            expect="robot_observations",
                        )
                        recv_observations: MsgReceived[RobotObservations] = await loop.run_in_executor(
                            executor, f
                        )
                        ro: RobotObservations = recv_observations.data
                        obs = cast(DB20Observations, ro.observations)
                        await webserver.push(f"{agent_name}-camera", obs.camera.jpg_data)

                    with tt.measure(f"agent_compute-{agent_name}"):
                        try:
                            map_data = cast(str, scenario.environment)
                            obs_plus = DB20ObservationsPlusState(
                                camera=obs.camera,
                                odometry=obs.odometry,
                                your_name=agent_name,
                                state=state_dump.data.state,
                                map_data=map_data,
                            )
                            # logger.info("sending agent", obs_plus=obs_plus)
                            agent_ci.write_topic_and_expect_zero("observations", obs_plus)
                            get_commands = GetCommands(t_effective)
                            f = functools.partial(
                                agent_ci.write_topic_and_expect,
                                "get_commands",
                                get_commands,
                                expect="commands",
                            )
                            r: MsgReceived = await loop.run_in_executor(executor, f)

                        except BaseException as e:
                            msg = "Trouble with communication to the agent."
                            raise dc.InvalidSubmission(msg) from e

                    with tt.measure("set_robot_commands"):
                        set_robot_commands = SetRobotCommands(agent_name, t_effective, r.data,)
                        f = functools.partial(
                            sim_ci.write_topic_and_expect_zero, "set_robot_commands", set_robot_commands,
                        )
                        await loop.run_in_executor(executor, f)

                # these are the *non* playing (above we do the playable)
                for npc_robot_name in not_playable_robots:
                    with tt.measure(f"sim_compute_robot_state-{npc_robot_name}"):
                        rs = GetRobotState(npc_robot_name, t_effective)
                        f = functools.partial(
                            sim_ci.write_topic_and_expect, "get_robot_state", rs, expect="robot_state",
                        )
                        await loop.run_in_executor(executor, f)

                with tt.measure("sim_compute_sim_state"):
                    f = functools.partial(sim_ci.write_topic_and_expect, "get_sim_state", expect="sim_state")
                    recv: MsgReceived[SimulationState] = await loop.run_in_executor(executor, f)

                    sim_state: SimulationState = recv.data

                    if sim_state.done:
                        if True:
                            msg = (
                                f"Breaking because of simulator ({sim_state.done_code} - "
                                f"{sim_state.done_why}"
                            )
                            logger.info(msg)
                            break
                        # else:
                        #     msg = "Not interrupting even though simulator wants "
                        #     logger.info(msg, sim_state=sim_state)

                with tt.measure("step_physics"):
                    current_sim_time += physics_dt
                    f = functools.partial(sim_ci.write_topic_and_expect_zero, "step", Step(current_sim_time))
                    await loop.run_in_executor(executor, f)

                with tt.measure("get_ui_image"):
                    f = functools.partial(
                        sim_ci.write_topic_and_expect, "get_ui_image", None, expect="ui_image"
                    )
                    r_ui_image: MsgReceived[JPGImage] = await loop.run_in_executor(executor, f)

            if True:
                await webserver.push("ui_image", r_ui_image.data.jpg_data)
                await asyncio.sleep(0.05)
            log_timing_info(tt, sim_ci)

    return current_sim_time


def log_timing_info(tt, sim_ci: ComponentInterface):
    ipce = ipce_from_object(tt)
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

    logger.info(f"Simulation provides observations {type_observations_sim}")
    logger.info(f"Simulation requires commands {type_commands_sim}")

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
        msg = f'Could not find variable "{name}"; I know:\n{json.dumps(environment, indent=4)}'
        raise Exception(msg)
    v = environment[name]
    try:
        return yaml.load(v, Loader=yaml.SafeLoader)
    except Exception as e:
        msg = f"Could not load YAML: {e}\n\n{v}"
        raise Exception(msg)


import duckietown_challenges as dc


def wrap(cie: dc.ChallengeInterfaceEvaluator) -> None:
    d = cie.get_tmp_dir()

    logdir = os.path.join(d, "episodes")

    attempts = os.path.join(d, "attempts")
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    if not os.path.exists(attempts):
        os.makedirs(attempts)
    try:
        asyncio.run(main(cie, logdir, attempts), debug=True)
        cie.set_score("simulation-passed", 1)
    except:
        cie.error(f"weird exception: {traceback.format_exc()}")
    finally:
        cie.info("saving files")
        cie.set_evaluation_dir("episodes", logdir)

    cie.info("experiment_manager::wrap() terminated gracefully.")


def go():
    with dc.scoring_context() as cie:
        try:
            wrap(cie)
        except RemoteNodeAborted as e:
            msg = (
                "It appears that one of the remote nodes has aborted.\n"
                "I will wait 10 seconds before aborting myself so that its\n"
                "error will be detected by the evaluator rather than mine."
            )
            msg += f"\n\n{traceback.format_exc()}"
            cie.error(msg)
            time.sleep(10)
            raise


if __name__ == "__main__":
    go()
