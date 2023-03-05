from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from .schemas import Offer, VideoTransformTrack


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


peerConnections = set()


@app.post('/offer')
async def offer(params: Offer):

    offer = RTCSessionDescription(sdp=params.sdp, type=params.type)
    peerConnection = RTCPeerConnection()
    peerConnections.add(peerConnection)
    recorder = MediaBlackhole()

    @peerConnection.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        if peerConnection.iceConnectionState == "failed":
            await peerConnection.close()
            peerConnections.discard(peerConnection)


    @peerConnection.on("track")
    def on_track(track):
        if track.kind == "audio":
            recorder.addTrack(track)
        elif track.kind == "video":
            local_video = VideoTransformTrack(
                track,
                params.video_transform
            )
            peerConnection.addTrack(local_video)

        @track.on("ended")
        async def on_ended():
            await recorder.stop()

    # Get Offer
    await peerConnection.setRemoteDescription(offer)
    await recorder.start()

    # Create Answer
    answer = await peerConnection.createAnswer()
    await peerConnection.setLocalDescription(answer)
    response = {"sdp": peerConnection.localDescription.sdp, "type": peerConnection.localDescription.type}

    # Send Answer
    return response

@app.on_event('shutdown')
async def on_shutdown(app):
    # Close peer connections...
    coros = [peerConnection.close() for peerConnection in peerConnections]
    await asyncio.gather(*coros)
    peerConnections.clear()

