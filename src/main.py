# Copyright (c) farm-ng, inc.
#
# Licensed under the Amiga Development Kit License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/farm-ng/amiga-dev-kit/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import cv2
import argparse
import asyncio
import io
import os
import numpy as np

from typing import List
from pathlib import Path
from cv2.ximgproc import anisotropicDiffusion

import grpc
from farm_ng.oak import oak_pb2
from farm_ng.oak.camera_client import OakCameraClient
from farm_ng.service import service_pb2
from farm_ng.service.service_client import ClientConfig
# from turbojpeg import TurboJPEG

os.environ["KIVY_NO_ARGS"] = "1"


from kivy.config import Config  # noreorder # noqa: E402

Config.set("graphics", "resizable", False)
Config.set("graphics", "width", "1280")
Config.set("graphics", "height", "800")
Config.set("graphics", "fullscreen", "false")
Config.set("input", "mouse", "mouse,disable_on_activity")
Config.set("kivy", "keyboard_mode", "systemanddock")

from kivy.app import App  # noqa: E402
from kivy.lang.builder import Builder  # noqa: E402
from kivy.graphics.texture import Texture  # noqa: E402
from kivy.core.window import Window  # noqa: E402
from kivy.input.providers.mouse import MouseMotionEvent  # noqa: E402

from kivy.uix.tabbedpanel import TabbedPanel


class CameraApp(App):
    def __init__(self, address: str, port: int, stream_every_n: int) -> None:
        super().__init__()
        self.address = address
        self.port = port
        self.stream_every_n = stream_every_n
        self.touchCnt=0
        self.firstPos = (10,10)
        self.secondPos = (50,50)

        # self.image_decoder = TurboJPEG()
        self.tasks: List[asyncio.Task] = []

    def build(self):
        
        def on_touch_down(window: Window, touch: MouseMotionEvent) -> bool:
            """Handles initial press with mouse click or touchscreen."""			
            if isinstance(touch, MouseMotionEvent) and int(
                os.environ.get("DISABLE_KIVY_OUSE_EVENTS", 0)
            ):		  
                return True
                
            for w in window.children[:]:
                if w.dispatch("on_touch_down", touch):
                    return True			 
                                      
            #print(self.root.current_tab.text)
                
            
            if self.root.ids['rgb'].collide_point(*touch.pos):		   
                # if the active tab is "Rgb" then process the touch wuthout this
                # there is a crossover with touch events on images of other tabs 
                # (probably due to the overalap of coords)
                if (self.root.ids["mainTab"].current_tab.text == "Rgb"):	 
                    # The touch has occurred inside the widgets area. Do stuff!
                    sizeXim=float(self.root.ids['rgb'].size[0])
                    normsizeXim=float(self.root.ids['rgb'].norm_image_size[0])
                    sizeYim=float(self.root.ids['rgb'].size[1])
                    normsizeYim=float(self.root.ids['rgb'].norm_image_size[1])
                    x0=(sizeXim-normsizeXim)/2.0
                    y0=(sizeYim-normsizeYim)/2.0
                    minX=min(sizeXim,normsizeXim)
                    minY=min(sizeYim,normsizeYim)
                    posTmp=((float(touch.pos[0])-x0)/minX,(float(touch.pos[1])-y0)/minY)
                    if (self.touchCnt==0):
                        self.firstPos=posTmp
                        self.touchCnt=self.touchCnt+1
                    elif (self.touchCnt==1): 
                        self.secondPos=posTmp
                        self.touchCnt=self.touchCnt+1
                    else:
                        self.touchCnt=0
                pass					   
                
            # Add additional on_touch_down behavior here
            return False

        Window.bind(on_touch_down=on_touch_down)

        return Builder.load_file("res/main.kv")

    def on_exit_btn(self) -> None:
        """Kills the running kivy application."""
        App.get_running_app().stop()

    async def app_func(self):
        async def run_wrapper():
            # we don't actually need to set asyncio as the lib because it is
            # the default, but it doesn't hurt to be explicit
            await self.async_run(async_lib="asyncio")
            for task in self.tasks:
                task.cancel()

        # configure the camera client
        config = ClientConfig(address=self.address, port=self.port)
        client = OakCameraClient(config)

        # Stream camera frames
        self.tasks.append(asyncio.ensure_future(self.stream_camera(client)))

        return await asyncio.gather(run_wrapper(), *self.tasks)

    async def stream_camera(self, client: OakCameraClient) -> None:
        """This task listens to the camera client's stream and populates the tabbed panel with all 4 image streams
        from the oak camera."""

        frameCount = 0
        fiveCount = 0
        fiveAverage = 0
        pixelsToMM = 10
        frameNumberCoutner = 4
        pMili = 0
        pArray = np.array([0,0])

        while self.root is None:
            await asyncio.sleep(0.01)

        response_stream = None

        while True:
            # check the state of the service
            state = await client.get_state()

            if state.value not in [
                service_pb2.ServiceState.IDLE,
                service_pb2.ServiceState.RUNNING,
            ]:
                # Cancel existing stream, if it exists
                if response_stream is not None:
                    response_stream.cancel()
                    response_stream = None
                print("Camera service is not streaming or ready to stream")
                await asyncio.sleep(0.1)
                continue

            # Create the stream
            if response_stream is None:
                response_stream = client.stream_frames(every_n=self.stream_every_n)

            try:
                # try/except so app doesn't crash on killed service
                response: oak_pb2.StreamFramesReply = await response_stream.read()
                assert response and response != grpc.aio.EOF, "End of stream"
            except Exception as e:
                print(e)
                response_stream.cancel()
                response_stream = None
                continue

            # get the sync frame
            # not the same frame that we're using
            frame: oak_pb2.OakSyncFrame = response.frame

            frameCount += 1
            
            if fiveCount <= frameNumberCoutner :
                fiveCount += 1
            else:
                fiveCount = 0

            # get image and show
            # for view_name in ["rgb", "disparity", "left", "right"]:
                # Skip if view_name was not included in frame
            try:
                # Decode the image and render it in the correct kivy texture
                # buf = np.frombuffer(getattr(frame, view_name).image_data, dtype=np.uint8)
                buf = np.frombuffer(getattr(frame, "rgb").image_data, dtype=np.uint8)
                img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
                # print(getattr(frame, view_name))

                # maybe change later
                '''
                texture = Texture.create(
                    size=(img.shape[1], img.shape[0]), icolorfmt="bgr"
                )
                texture.flip_vertical()
                texture.blit_buffer(
                    img.tobytes(),
                    colorfmt="bgr",
                    bufferfmt="ubyte",
                    mipmap_generation=False,
                )
                # self.root.ids[view_name].texture = texture
                self.root.ids["rgb"].texture = texture
                '''


            except Exception as e:
                print(e)
                continue
            
            rgb = img

            for view_name in ["rgb", "cropped","control","errorGraph"]:

                # rgb = frame
                rgb = cv2.flip(rgb,0)
                color=(0,0,255)
                start_point = (int(self.firstPos[0]*rgb.shape[1]), 
                    int(self.firstPos[1]*rgb.shape[0]))
                end_point = (int(self.secondPos[0]*rgb.shape[1]), 
                    int(self.secondPos[1]*rgb.shape[0]))	
                    #print(self.touchCnt,start_point,end_point) 

                    # if the view is rgb image
                if (view_name == "rgb"):
                    if (self.touchCnt == 2):
                        rgb = cv2.rectangle(rgb, start_point, end_point, color, 10)		   
                    data = rgb.tobytes()
                    texture = Texture.create(size=(rgb.shape[1],rgb.shape[0]), colorfmt="bgr")
                    texture.blit_buffer(data, bufferfmt="ubyte", colorfmt="bgr")
                    self.root.ids[view_name].texture = texture
                    # the view is cropped image
                elif view_name == "cropped":
                    if (self.touchCnt == 2):
                        if (start_point[0] > end_point[0]):
                            tmp=int(start_point[0])
                            tmp1=int(end_point[0])
                            start_point=(tmp1, start_point[1])
                            end_point=(tmp, end_point[1])
                        if (start_point[1] > end_point[1]):
                            tmp=int(start_point[1])
                            tmp1=int(end_point[1])
                            start_point=(start_point[0],tmp1)
                            end_point=(end_point[0],tmp)
                        crgb = rgb[start_point[1]:end_point[1],start_point[0]:end_point[0]]
                            
                            # ------------------------------------------------------------- 
                            # ---->  image processing of the cropped image in openCV <----
                            # ------------------------------------------------------------- 
                        
                        hsv = cv2.cvtColor(crgb, cv2.COLOR_RGB2HSV)

                        low_red = np.array([0, self.root.ids.slider_LR.value, 0])               
                        low_green = np.array([60, self.root.ids.slider_LG.value, 0])
                        low_blue = np.array([120, self.root.ids.slider_LB.value, 0])

                        high_red  = np.array([30, self.root.ids.slider_HR.value, 255])
                        high_green = np.array([90, self.root.ids.slider_HG.value, 255])
                        high_blue = np.array([150, self.root.ids.slider_HB.value, 255])

                        green_mask = cv2.inRange(hsv, low_green, high_green)
                        red_mask = cv2.inRange(hsv, low_red, high_red)
                        blue_mask = cv2.inRange(hsv, low_blue, high_blue)
                        
                        mask = np.add(green_mask, red_mask)
                        mask2 = np.add(mask, blue_mask)
                        
                        ## ftd = filtered image
                        ftd = cv2.bitwise_and(crgb, crgb, mask=mask2)

                        ## diffusion of image k = iterations
                        k = 100
                        diffused = anisotropicDiffusion(ftd,0.075 ,20, k)
                        ## thresholding to clean diffused image for clustering

                        ret, pcrgb = cv2.threshold(diffused, 75, 255, cv2.THRESH_TOZERO)

                        grey = cv2.cvtColor(pcrgb, cv2.COLOR_BGR2GRAY)

                        contours, hierarchy = cv2.findContours(grey, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cont = cv2.drawContours(crgb, contours, -1, (0, 0, 255), 3)

                        circleImg = cont.copy()
                        circleInfo = np.zeros([len(contours),3])

                        lowRadius = self.root.ids.lowRadiusSlider.value
                        highRadius = self.root.ids.highRadiusSlider.value

                        for i in range(len(contours)):
                            cnt = contours[i]
                            (x,y),radius = cv2.minEnclosingCircle(cnt)
                            if (radius > lowRadius) and (radius < highRadius):
                                circleInfo[i,0] = int(x)
                                circleInfo[i,1] = int(y)
                                circleInfo[i,2] = int(radius)
                                center = (int(x),int(y))
                                radius = int(radius)
                                circleImg = cv2.circle(circleImg,center,radius,(255,0,0),3)

                        circleInfo = circleInfo[~np.all(circleInfo == 0, axis=1)]

                        sortedByRadius = circleInfo[circleInfo[:,2].argsort()]

                        fewCircles = cont.copy()

                        toolWidth = int(fewCircles.shape[1])
                        imageHeight = int(fewCircles.shape[0])

                        centerPoint = int(toolWidth/2)
                        centerLine = cv2.line(fewCircles, (centerPoint,0), (centerPoint,imageHeight), (0,255,255), 2)

                        sumError = 0

                        correctRadius = np.copy(sortedByRadius)

                        for i in range(len(correctRadius[:,0])):
                            center = (int(correctRadius[i,0]),int(correctRadius[i,1]))
                            radius = int(correctRadius[i,2])
                            fewCircles = cv2.circle(fewCircles,center,radius,(255,0,0),3)
                            fewCircles = cv2.line(fewCircles, (int(correctRadius[i,0]),int(correctRadius[i,1])), (centerPoint,int(correctRadius[i,1])), (255,0,255), 2)
                            sumError += centerPoint - int(correctRadius[i,0])
                                                        
                        scale_percent = 200 # percent of original size
                        width = int(centerLine.shape[1] * scale_percent / 100)
                        height = int(centerLine.shape[0] * scale_percent / 100)
                        dim = (width, height)
                        resizedCenterLine = cv2.resize(centerLine, dim, interpolation = cv2.INTER_AREA)
                                                        
                        if len(correctRadius[:,0]) == 0:
                            pFrame = 0
                        else:
                            pFrame = sumError / len(correctRadius[:,0])

                        edge_flip = cv2.flip(fewCircles, 0)
                        #cv2.imshow('CenterLine',fewCircles)
                        cv2.waitKey(10)
                        
                        if fiveCount <= frameNumberCoutner :
                            fiveAverage += pFrame
                            #print("Five party!!!!!!!!!!!!!", fiveCount)
                        elif fiveCount > frameNumberCoutner :

                            finalFiveAverage = fiveAverage/5

                            pMili = finalFiveAverage/pixelsToMM
                            #print("pMILI PMILI PMILI PMILI PMILI",pMili)
                            fiveAverage = 0
                            finalFiveAverage = 0
                            """
                            P mili is the average of the last five iamges. 
                            can reset pMili with this
                            pMili = 0
                            """
                            
                            
                        print("Average of last 5 :--- --- ", pMili)
                        pArray = np.vstack([pArray, [frameCount, pMili]])
                        
                        x = pArray[:, 0]
                        p = pArray[:, 1]
                        

                        # Add axis labels
                        '''
                        plt.title("Average error over 5 frames")
                        plt.xlabel("Frame Count")
                        plt.ylabel("Error (pixles/meter)")

                        plt.plot(x, p)
                        plt.savefig("errorPlotImage.png")

                        errorGraph = Image(source="errorPlotImage.png")
                        
                        # Show the plot

                        #edge = cv2.Canny(pcrgb, 0, 100, L2gradient = True)
                        #edge_flip = cv2.flip(edge, 0)
                        #cv2.imshow('Edge', edge_flip)
                        #cv2.waitKey(10)
                        #pcrgb = thresh
                        '''
                        # --------------------------------------------------------------       
                        # --- end of imagen processing og the cropped image in openCV ---
                        # --------------------------------------------------------------
                            
                        # update images
                        data = crgb.tobytes()
                        texture = Texture.create(size=(crgb.shape[1],crgb.shape[0]), colorfmt="bgr")
                        texture.blit_buffer(data, bufferfmt="ubyte", colorfmt="bgr")
                        pdata= pcrgb.tobytes()
                        ptexture = Texture.create(size=(pcrgb.shape[1],pcrgb.shape[0]), colorfmt="bgr")							 
                        ptexture.blit_buffer(pdata, bufferfmt="ubyte", colorfmt="bgr")
                    
                        cdata = resizedCenterLine.tobytes()
                        ctexture = Texture.create(size=(resizedCenterLine.shape[1],resizedCenterLine.shape[0]), colorfmt="bgr")	
                        ctexture.blit_buffer(cdata, bufferfmt="ubyte", colorfmt="bgr")
                    
                    
                        self.root.ids["cropImage"].texture = texture
                        self.root.ids["procImage"].texture = ptexture
                        self.root.ids["controlImage"].texture = ctexture

                    else:
                        data = rgb.tobytes()
                        pcrgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
                        pdata= pcrgb.tobytes()
                        texture = Texture.create(size=(rgb.shape[1],rgb.shape[0]), colorfmt="bgr")
                        ptexture = Texture.create(size=(pcrgb.shape[1],pcrgb.shape[0]), colorfmt="bgr")
                        texture.blit_buffer(data, bufferfmt="ubyte", colorfmt="bgr")
                        ptexture.blit_buffer(pdata, bufferfmt="ubyte", colorfmt="luminance")
                        self.root.ids["cropImage"].texture = texture
                        self.root.ids["procImage"].texture = ptexture																			 
                # all other views 
                else:
                    data = rgb.tobytes()
                    texture = Texture.create(size=(rgb.shape[1],rgb.shape[0]), colorfmt="bgr")
                    texture.blit_buffer(data, bufferfmt="ubyte", colorfmt="bgr")
                    self.root.ids[view_name].texture = texture            

            


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="amiga-camera-app")
    parser.add_argument("--port", type=int, required=True, help="The camera port.")
    parser.add_argument(
        "--address", type=str, default="localhost", help="The camera address"
    )
    parser.add_argument(
        "--stream-every-n", type=int, default=1, help="Streaming frequency"
    )
    args = parser.parse_args()

    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(
            CameraApp(args.address, args.port, args.stream_every_n).app_func()
        )
    except asyncio.CancelledError:
        pass
    loop.close()