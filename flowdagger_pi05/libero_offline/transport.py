"""Optional transport for long, blocking local simulator/model operations."""
from openpi.serving.websocket_policy_server import WebsocketPolicyServer, _health_check
import websockets.asyncio.server as server


class NoPingPolicyServer(WebsocketPolicyServer):
    async def run(self):
        async with server.serve(
            self._handler, self._host, self._port, compression=None, max_size=None,
            process_request=_health_check, ping_interval=None, ping_timeout=None,
        ) as service:
            await service.serve_forever()
