import os

from unitree_webrtc_connect.webrtc_driver import (
    UnitreeWebRTCConnection,
    WebRTCConnectionMethod,
)


class Go2Connection:
    """Go2 WebRTC 연결을 관리하는 클래스."""

    def __init__(self, ip=None):
        self.ip = ip or "192.168.0.101"
        self.conn = None

    async def connect(self):
        if not self.ip:
            raise ValueError("Go2 IP가 설정되지 않았습니다.")

        # Go2 firmware 1.1.15+ requires a per-device AES-128 key.
        aes_key = os.environ.get("GO2_AES_KEY")

        if not aes_key:
            raise ValueError(
                "GO2_AES_KEY 환경변수가 설정되지 않았습니다."
            )

        print(f"[GO2] Connecting to {self.ip}...")

        self.conn = UnitreeWebRTCConnection(
            WebRTCConnectionMethod.LocalSTA,
            ip=self.ip,
            aes_128_key=aes_key,
        )

        await self.conn.connect()

        print("[GO2] Connected successfully.")
        return self.conn

    def get_connection(self):
        if self.conn is None:
            raise RuntimeError("Go2가 아직 연결되지 않았습니다.")

        return self.conn