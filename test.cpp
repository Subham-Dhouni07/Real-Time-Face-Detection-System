#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;


int main() {
    // Create a VideoCapture object to access the webcam
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Error: Unable to access the webcam." <<endl;
        return -1;
    }

    // Load the pre-trained face detection model
    CascadeClassifier faceCascade;
    if (!faceCascade.load("Resources/haarcascade_frontalface_default.xml")) {
        cerr << "Error: Unable to load the face detection model." <<endl;
        return -1;
    }

    while (true) {
        Mat frame;
        cap.read(frame);

        if (frame.empty()) {
            cerr << "Error: Unable to read frame from the webcam." << endl;
            break;
        }

        // Convert frame to grayscale for face detection
        Mat grayFrame;
        cvtColor(frame, grayFrame, COLOR_BGR2GRAY);

        // Detect faces in the frame
        vector<Rect> faces;
        faceCascade.detectMultiScale(grayFrame, faces, 1.1, 3, 0, Size(30, 30));

        // Draw rectangles around the detected faces
        for (const auto& face : faces) {
            rectangle(frame, face, Scalar(0, 255, 0), 2);
        }

        // Display the processed frame with detected faces
        imshow("Face Detection", frame);

        // Break the loop if the user presses the 'Esc' key
        if (waitKey(1) == 27) {
            break;
        }
    }

    // Release the VideoCapture and close the window
    cap.release();
    destroyAllWindows();

    return 0;
}
