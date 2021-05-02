#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/tracking/tracking_legacy.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <chrono>
#include <windows.h>
#include <EvColl.h>
#include <winbase.h>

using namespace cv;
using namespace std;

const string list_filename = "test_files_list.txt"; //"e:\\OpenCV\\Tracking\\TrackerTest\\x64\\Debug\\test_files_list.txt";
#define KCF_COLOR               Scalar(255, 0, 0)
#define MIL_COLOR               Scalar(0, 255, 0)
#define GOTURN_COLOR            Scalar(0, 0, 255)
#define CSRT_COLOR              Scalar(255, 0, 255)

#define BOOSTING_COLOR          Scalar(255, 191, 0)
#define MOSSE_COLOR             Scalar(255, 255, 0)
#define TLD_COLOR               Scalar(0, 191, 255)
#define MEDIANFLOW_COLOR        Scalar(0, 255, 255)

#define SHOW_DISPLAY            1
/* Mode 1 - all-in-one
*  Mode 2 - video grid MxN
*  Mode 3 - PIP
*/
#define MODE                    1 

struct Task
{
    string input_file;
    string output_file;
    int start_frame;
    int end_frame;
    int roi_pos_x;
    int roi_pos_y;
    int roi_size_width;
    int roi_size_height;
};

template<typename T, typename T1>
int updateFrame(T& tracker, Mat& src, T1& roi)
{
    std::chrono::high_resolution_clock::time_point t_start, t_finish;
    t_start = std::chrono::high_resolution_clock::now();
    tracker->update(src, roi);
    t_finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> delta_millis = t_finish - t_start;
    return static_cast<int>(delta_millis.count());
}

int main(int argc, char** argv)
{
    std::vector <Task> tasks;

    if (argc != 2)
    {
        cout << " Usage: " << argv[0] << " File path to testing tasks list " << endl;
        cout << " Default using \"test_files_list.txt\" " << endl;
    }

    std::string line;
    std::ifstream infile(list_filename);

    while (std::getline(infile, line))
    {
        std::istringstream iss(line);
        Task tmp;
        if (!(iss >> tmp.input_file >> tmp.output_file >> tmp.start_frame >> tmp.end_frame >> tmp.roi_pos_x >> tmp.roi_pos_y >> tmp.roi_size_width >> tmp.roi_size_height))
        { 
            return -1; 
        } // error

        tasks.push_back(tmp);
    }

    SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED);

    for (auto &t : tasks)
    {
        // Task cycle

        // Load file
        std::cout << t.input_file << "  " << t.output_file << std::endl;

        Mat src;
        // use default camera as video source
        VideoCapture cap(t.input_file);
        // ¬ыставл€ем на определенный кадр видеозаписи. 
        cap.set(CAP_PROP_POS_FRAMES, t.start_frame);

        // check if we succeeded
        if (!cap.isOpened()) {
            cerr << "ERROR! Unable to open camera\n";
            continue;
            //return -1;
        }
        // get one frame from camera to know frame size and type
        cap >> src;
        // check if we succeeded
        if (src.empty()) {
            cerr << "ERROR! blank frame grabbed\n";
            continue;
            //return -1;
        }
        bool isColor = (src.type() == CV_8UC3);

        //--- INITIALIZE VIDEOWRITER
        VideoWriter writer;
        int codec = VideoWriter::fourcc('M', 'J', 'P', 'G');  // select desired codec (must be available at runtime)
        double fps = cap.get(CAP_PROP_FPS);                          // framerate of the created video stream
       
        writer.open(t.output_file, codec, fps, src.size(), isColor);
        // check if we succeeded
        if (!writer.isOpened()) {
            cerr << "Could not open the output video file for write\n";
            continue;
            //return -1;
        }

        // declares all required variables
        Rect roi_kcf(t.roi_pos_x, t.roi_pos_y, t.roi_size_width, t.roi_size_height);
        Rect roi_mil(t.roi_pos_x, t.roi_pos_y, t.roi_size_width, t.roi_size_height);
        Rect roi_goturn(t.roi_pos_x, t.roi_pos_y, t.roi_size_width, t.roi_size_height);
        Rect roi_csrt(t.roi_pos_x, t.roi_pos_y, t.roi_size_width, t.roi_size_height);
        Rect2d roi_boosting(t.roi_pos_x, t.roi_pos_y, t.roi_size_width, t.roi_size_height);
        Rect2d roi_mosse(t.roi_pos_x, t.roi_pos_y, t.roi_size_width, t.roi_size_height);
        Rect2d roi_tld(t.roi_pos_x, t.roi_pos_y, t.roi_size_width, t.roi_size_height);
        Rect2d roi_medianflow(t.roi_pos_x, t.roi_pos_y, t.roi_size_width, t.roi_size_height);
        
        Mat frame_kcf;
        Mat frame_mil;
        Mat frame_goturn;
        Mat frame_csrt;
        Mat frame_boosting;
        Mat frame_mosse;
        Mat frame_tld;
        Mat frame_medianflow;

        Mat frame_all_in_one;

        //Copy frames to trackers
        src.copyTo(frame_kcf);
        src.copyTo(frame_mil);
        src.copyTo(frame_goturn);
        src.copyTo(frame_csrt);
        src.copyTo(frame_boosting);
        src.copyTo(frame_mosse);
        src.copyTo(frame_tld);
        src.copyTo(frame_medianflow);

        Ptr<Tracker> tracker_kcf = cv::TrackerKCF::create();
        Ptr<Tracker> tracker_mil = cv::TrackerMIL::create();
        Ptr<Tracker> tracker_goturn = cv::TrackerGOTURN::create();
        Ptr<Tracker> tracker_csrt = cv::TrackerCSRT::create();

        Ptr<legacy::Tracker> tracker_boosting = cv::legacy::TrackerBoosting::create();
        Ptr<legacy::Tracker> tracker_mosse = cv::legacy::TrackerMOSSE::create();
        Ptr<legacy::Tracker> tracker_tld = cv::legacy::TrackerTLD::create();
        Ptr<legacy::Tracker> tracker_medianflow = cv::legacy::TrackerMedianFlow::create();

        // initialize the tracker
        
        tracker_kcf->init(src, roi_kcf);
        tracker_mil->init(src, roi_mil);
        tracker_goturn->init(src, roi_goturn);
        tracker_csrt->init(frame_csrt, roi_csrt);
        
        tracker_boosting->init(frame_boosting, roi_boosting); 
        tracker_mosse->init(frame_mosse, roi_mosse);
        tracker_tld->init(frame_tld, roi_tld);
        tracker_medianflow->init(frame_medianflow, roi_medianflow);
        
        // perform the tracking process
        printf("Start the tracking process, press ESC to quit.\n");

        //--- GRAB AND WRITE LOOP
        cout << "Writing videofile: " << t.output_file << endl
            << "Press any key to terminate" << endl;

        float fps_kcf, fps_mil, fps_goturn, fps_csrt;
        float fps_boosting, fps_mosse, fps_tld, fps_medianflow;
        float frame_time_kcf = 0, frame_time_mil = 0, frame_time_goturn = 0, frame_time_csrt = 0;
        float frame_time_boosting = 0, frame_time_mosse = 0, frame_time_tld = 0, frame_time_medianflow = 0;

        for (int i = t.start_frame; i < (t.end_frame < t.start_frame ? cap.get(CAP_PROP_FRAME_COUNT) : t.end_frame); i++)
        {
            // check if we succeeded
            if (!cap.read(src)) {
                cerr << "ERROR! blank frame grabbed\n";
                continue;
            }
            //src.copyTo(frame_kcf);
            src.copyTo(frame_all_in_one);

            // update the tracking result
            // modern
            frame_time_kcf          = frame_time_kcf != 0        ? frame_time_kcf        * 0.95f + updateFrame(tracker_kcf, src, roi_kcf)               * 0.05f : updateFrame(tracker_kcf, src, roi_kcf);
            frame_time_mil          = frame_time_mil != 0        ? frame_time_mil        * 0.95f + updateFrame(tracker_mil, src, roi_mil)               * 0.05f : updateFrame(tracker_mil, src, roi_mil);
            frame_time_goturn       = frame_time_goturn != 0     ? frame_time_goturn     * 0.95f + updateFrame(tracker_goturn, src, roi_goturn)         * 0.05f : updateFrame(tracker_goturn, src, roi_goturn);
            frame_time_csrt         = frame_time_csrt != 0       ? frame_time_csrt       * 0.95f + updateFrame(tracker_csrt, src, roi_csrt)             * 0.05f : updateFrame(tracker_csrt, src, roi_csrt);
            //legacy
            frame_time_boosting     = frame_time_boosting != 0   ? frame_time_boosting   * 0.95f + updateFrame(tracker_boosting, src, roi_boosting)     * 0.05f : updateFrame(tracker_boosting, src, roi_boosting);
            frame_time_mosse        = frame_time_mosse != 0      ? frame_time_mosse      * 0.95f + updateFrame(tracker_mosse, src, roi_mosse)           * 0.05f : updateFrame(tracker_mosse, src, roi_mosse);
            frame_time_tld          = frame_time_tld != 0        ? frame_time_tld        * 0.95f + updateFrame(tracker_tld, src, roi_tld)               * 0.05f : updateFrame(tracker_tld, src, roi_tld);
            frame_time_medianflow   = frame_time_medianflow != 0 ? frame_time_medianflow * 0.95f + updateFrame(tracker_medianflow, src, roi_medianflow) * 0.05f : updateFrame(tracker_medianflow, src, roi_medianflow);


            // draw the tracked object
            // modern
            rectangle(frame_all_in_one, roi_kcf, KCF_COLOR, 2, 1);
            rectangle(frame_all_in_one, roi_mil, MIL_COLOR, 2, 1);
            rectangle(frame_all_in_one, roi_goturn, GOTURN_COLOR, 2, 1);
            rectangle(frame_all_in_one, roi_csrt, CSRT_COLOR, 2, 1);
            //legacy
            rectangle(frame_all_in_one, roi_boosting, BOOSTING_COLOR, 2, 1);
            rectangle(frame_all_in_one, roi_mosse, MOSSE_COLOR, 2, 1);
            rectangle(frame_all_in_one, roi_tld, TLD_COLOR, 2, 1);
            rectangle(frame_all_in_one, roi_medianflow, MEDIANFLOW_COLOR, 2, 1);

            // draw 
            char text_kcf[255];
            char text_mil[255];
            char text_goturn[255];
            char text_csrt[255];
            char text_boosting[255];
            char text_mosse[255];
            char text_tld[255];
            char text_medianflow[255];

            sprintf_s(text_kcf,         "KCF      aver. frame: %.1f ms  FPS: %.2f", frame_time_kcf, 1000.0f / frame_time_kcf);
            sprintf_s(text_mil,         "MIL       aver. frame: %.1f ms  FPS: %.2f", frame_time_mil, 1000.0f / frame_time_mil);
            sprintf_s(text_goturn,      "GOTURN  aver. frame: %.1f ms  FPS: %.2f", frame_time_goturn, 1000.0f / frame_time_goturn);
            sprintf_s(text_csrt,        "CSRT     aver. frame: %.1f ms  FPS: %.2f", frame_time_csrt, 1000.0f / frame_time_csrt);

            sprintf_s(text_boosting,    "BOOSTING    aver. frame: %.1f ms  FPS: %.2f", frame_time_boosting, 1000.0f / frame_time_boosting);
            sprintf_s(text_mosse,       "MOSSE       aver. frame: %.1f ms  FPS: %.2f", frame_time_mosse, 1000.0f / frame_time_mosse);
            sprintf_s(text_tld,         "TLD           aver. frame: %.1f ms  FPS: %.2f", frame_time_tld, 1000.0f / frame_time_tld);
            sprintf_s(text_medianflow,  "MEDIANFLOW  aver. frame: %.1f ms  FPS: %.2f", frame_time_medianflow, 1000.0f / frame_time_medianflow);

            // display FPS on frame
            putText(frame_all_in_one, text_kcf, Point(100, 40), FONT_HERSHEY_SIMPLEX, 0.75, KCF_COLOR, 2);
            putText(frame_all_in_one, text_mil, Point(100, 70), FONT_HERSHEY_SIMPLEX, 0.75, MIL_COLOR, 2);
            putText(frame_all_in_one, text_goturn, Point(100, 100), FONT_HERSHEY_SIMPLEX, 0.75, GOTURN_COLOR, 2);
            putText(frame_all_in_one, text_csrt, Point(100, 130), FONT_HERSHEY_SIMPLEX, 0.75, CSRT_COLOR, 2);

            putText(frame_all_in_one, text_boosting, Point(1060, 40), FONT_HERSHEY_SIMPLEX, 0.75, BOOSTING_COLOR, 2);
            putText(frame_all_in_one, text_mosse, Point(1060, 70), FONT_HERSHEY_SIMPLEX, 0.75, MOSSE_COLOR, 2);
            putText(frame_all_in_one, text_tld, Point(1060, 100), FONT_HERSHEY_SIMPLEX, 0.75, TLD_COLOR, 2);
            putText(frame_all_in_one, text_medianflow, Point(1060, 130), FONT_HERSHEY_SIMPLEX, 0.75, MEDIANFLOW_COLOR, 2);

            // show image with the tracked object
#ifdef SHOW_DISPLAY
            imshow("tracker", frame_all_in_one);
#endif
            // encode the frame into the videofile stream
            writer.write(frame_all_in_one);
            // show source video
            //imshow("Source", src);
            if (waitKey(5) == 27)
                return 0;

            src.copyTo(frame_kcf);

        }
        cap.release();
        writer.release();

    }
    SetThreadExecutionState(ES_CONTINUOUS);

    return 0;
}