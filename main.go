package main

import (
        "fmt"
        "image"
        "log"
        "math"
        "time"

        "gocv.io/x/gocv"
        "gocv.io/x/gocv/contrib"
        "gocv.io/x/gocv/contrib/mediapipe"
        "gocv.io/x/gocv/contrib/mediapipe/solutions"
        "github.com/go-vgo/robotgo"
)

func main() {
        webcam, err := gocv.OpenVideoCapture(0)
        if err != nil {
                log.Fatalf("Error opening video capture device: %v", err)
        }
        defer webcam.Close()

        window := gocv.NewWindow("Hand Tracking")
        defer window.Close()

        img := gocv.NewMat()
        defer img.Close()

        handDetector := mediapipe.NewHandDetector(
                solutions.NewHandDetectionOptions(),
        )
        defer handDetector.Close()

        handLandmarker := mediapipe.NewHandLandmarker(
                solutions.NewHandLandmarkerOptions(),
        )
        defer handLandmarker.Close()

        lastClickTime := time.Now()
        clickCooldown := 500 * time.Millisecond

        screenWidth, screenHeight := robotgo.GetScreenSize()

        // NPU Check and Model Loading
        useNPU := false
        net := gocv.Net{}
        if gocv.GetAvailableBackends(gocv.NetTargetVPU) != nil { //check for VPU
                net = gocv.ReadNetFromModelOptimizer("hand_detection.xml", "hand_detection.bin") // Replace with your model files
                if !net.Empty() {
                        net.SetPreferableBackend(gocv.NetBackendOpenVINO)
                        net.SetPreferableTarget(gocv.NetTargetVPU)
                        useNPU = true
                        log.Println("NPU Detected and Enabled")
                } else {
                        log.Println("NPU Detected, but model loading failed, falling back to CPU")
                }

        } else {
                log.Println("NPU not detected, running on CPU")
        }

        for {
                if ok := webcam.Read(&img); !ok {
                        log.Printf("Cannot read device %d", 0)
                        continue
                }
                if img.Empty() {
                        continue
                }

                rgb := gocv.NewMat()
                defer rgb.Close()
                gocv.CvtColor(img, &rgb, gocv.ColorBGRToRGB)

                var detections []mediapipe.Detection
                var landmarks []mediapipe.NormalizedLandmarkList

                if useNPU {
                        blob := gocv.BlobFromImage(rgb, 1.0, image.Pt(300, 300), gocv.NewScalar(0, 0, 0, 0), false, false)
                        net.SetInput(blob, "")
                        output := net.Forward("")
                        //Process NPU output to get detections and landmarks. (This is complex and depends on the model)
                        //This part needs to be implemented.
                        //Note that MediaPipe models are hard to convert to openvino, and this part requires deep knowledge of both.
                        log.Println("NPU inference is not implemented in this example")
                        detections = handDetector.Process(rgb)
                        landmarks = handLandmarker.Process(rgb)
                } else {
                        detections = handDetector.Process(rgb)
                        landmarks = handLandmarker.Process(rgb)
                }

                if len(landmarks) > 0 {
                        landmarkList := landmarks[0].Landmark

                        if len(landmarkList) > 8 {
                                x := int(landmarkList[8].X * float32(img.Cols()))
                                y := int(landmarkList[8].Y * float32(img.Rows()))

                                screenX := int(float64(x) * (float64(screenWidth) / float64(img.Cols())))
                                screenY := int(float64(y) * (float64(screenHeight) / float64(img.Rows())))

                                robotgo.MoveMouse(screenX, screenY)

                                thumbTip := landmarkList[4]
                                indexTip := landmarkList[8]
                                distance := calculateDistance(thumbTip, indexTip, img.Cols(), img.Rows())

                                if distance < 50 {
                                        if time.Since(lastClickTime) > clickCooldown {
                                                robotgo.Click("left")
                                                lastClickTime = time.Now()
                                        }
                                }
                        }

                        for _, landmark := range landmarkList {
                                px := int(landmark.X * float32(img.Cols()))
                                py := int(landmark.Y * float32(img.Rows()))
                                gocv.Circle(&img, image.Point{px, py}, 5, image.Scalar{0, 255, 0, 0}, 2)
                        }
                }

                window.IMShow(img)
                if window.WaitKey(1) >= 0 {
                        break
                }
        }
}

func calculateDistance(p1, p2 mediapipe.NormalizedLandmark, width, height int) float64 {
        x1 := float64(p1.X * float32(width))
        y1 := float64(p1.Y * float32(height))
        x2 := float64(p2.X * float32(width))
        y2 := float64(p2.Y * float32(height))
        return math.Sqrt(math.Pow(x2-x1, 2) + math.Pow(y2-y1, 2))
}
