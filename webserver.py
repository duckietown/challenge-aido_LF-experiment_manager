import asyncio
from asyncio import CancelledError
from typing import Dict

from aiohttp import MultipartWriter, web
from aiohttp.abc import Request
from multidict import CIMultiDict


class WebServer:
    def __init__(self, address: str, port: int):
        self.address = address
        self.port = port
        self.name2sub2queue: Dict[str, Dict[str, asyncio.Queue[bytes]]] = {}
        self.i = 0

    async def push(self, name: str, jpg_data: bytes):
        if not name in self.name2sub2queue:
            self.name2sub2queue[name] = {}
        for k, v in self.name2sub2queue[name].items():
            v.put_nowait(jpg_data)

    async def init(self):
        app = web.Application()
        app.router.add_route("GET", "/", self.index)
        app.router.add_route("GET", "/image", self.mjpeg_handler)
        loop = asyncio.get_event_loop()
        await loop.create_server(app.make_handler(), self.address, self.port)

    async def mjpeg_handler(self, request: Request):
        i = self.i
        self.i += 1

        subname = f"sub{i}"
        image_name = request.query["image"]
        queue: "asyncio.Queue[bytes]" = asyncio.Queue()
        self.name2sub2queue[image_name][subname] = queue
        my_boundary = "jpegboundary"
        response = web.StreamResponse(
            status=200,
            reason="OK",
            headers={"Content-Type": "multipart/x-mixed-replace;" "boundary=%s" % my_boundary,},
        )
        await response.prepare(request)

        try:
            while True:
                jpg_data = await queue.get()
                with MultipartWriter("x-mixed-replace", boundary=my_boundary) as mpwriter:
                    headers = CIMultiDict()
                    mpwriter.append(jpg_data, headers=headers)
                    await mpwriter.write(response, close_boundary=False)

                await asyncio.sleep(0)
        except CancelledError:
            raise
        finally:
            pass  # TODO: close?

    async def index(self, request):
        response = """
        
        <html>
        <head></head>
        <body>
        
        
        """
        for image_name in self.name2sub2queue:
            response += f'\n<img width="320" src="/image?image={image_name}"/>'

        response += "</body></html>"
        return web.Response(text=response, content_type="text/html")
