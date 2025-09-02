//  .cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//
#include <string>
#include <unistd.h>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "ModelRenderScene.h"
#include "EarthWalkManipulator.h"
#include "Log.h"
using namespace std;
using namespace cv;

void dumpMatrix4D(osg::Matrixd &matrix)
{
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            cout << fixed << setw(8) << setfill('0') << matrix(i, j) << "\t";
        }
        cout << endl; // 每行结束换行
    }
}

int main(int argc, char *argv[])
{
    bool showImages = false;
    bool saveImages = false;
    std::string pathModel;
    std::string logFile;
    std::string logServerity;
    bool saveLog = false;
    osg::NotifySeverity notifyServerity = osg::INFO;
    uint32_t viewWidth = 800, viewHeight = 600;
    std::string strWidth = "";
    std::string strHeight = "";
    // use an ArgumentParser object to manage the program arguments.
    osg::ArgumentParser arguments(&argc, argv);

    arguments.getApplicationUsage()->setApplicationName(arguments.getApplicationName());
    arguments.getApplicationUsage()->setDescription(arguments.getApplicationName() +
                                                    " Render the Model, capture the images(color image and depth image), "
                                                    "and then transfer to opencv for processing.");

    arguments.getApplicationUsage()->addCommandLineOption("-h or --help", "List command line options.");
    arguments.getApplicationUsage()->addCommandLineOption("--window <width> <height>", "Create a view of specified dimensions.");
    arguments.getApplicationUsage()->addCommandLineOption("--path <pathModel>", "The path of model.");
    arguments.getApplicationUsage()->addCommandLineOption("--save", "Save the color image and depth image.");
    arguments.getApplicationUsage()->addCommandLineOption("--show", "Show the color image and depth image.");
    arguments.getApplicationUsage()->addCommandLineOption("--log <filename>", "Save log to file.");
    // arguments.getApplicationUsage()->addCommandLineOption("--log
    //                                                           void dumpMatrix4D(osg::Matrixd & matrix) {
    //                                                               for (int i = 0; i < 4; i++)
    //                                                               {
    //                                                                   for (int j = 0; j < 4; j++)
    //                                                                   {
    //                                                                       cout << fixed << setw(8) << setfill('0') << matrix(i, j) << "\t";
    //                                                                   }
    //                                                                   cout << endl; // 每行结束换行
    //                                                               }
    //                                                           } Serverity < serverity > ", " Set the log serverity :\n "
    //                                                                                                                    "0: ALWAYS; 1: FATAL; 2: WARN; 3: NOTICE; \n"
    //                                                                                                                    "4: INFO; 5: DEBUG_INFO; 6: DEBUG_FP");

    if (arguments.read("-h") || arguments.read("--help"))
    {
        arguments.getApplicationUsage()->write(std::cout, osg::ApplicationUsage::COMMAND_LINE_OPTION);
        return 1;
    }
    if (arguments.errors())
    {
        arguments.writeErrorMessages(std::cout);
        return 1;
    }
    if (arguments.argc() <= 1)
    {
        arguments.getApplicationUsage()->write(std::cout, osg::ApplicationUsage::COMMAND_LINE_OPTION);
        return 1;
        void dumpMatrix4D(osg::Matrixd & matrix)
        {
            for (int i = 0; i < 4; i++)
            {
                for (int j = 0; j < 4; j++)
                {
                    cout << fixed << setw(8) << setfill('0') << matrix(i, j) << "\t";
                }
                cout << endl; // 每行结束换行
            }
        };
        while (arguments.read("--save"))
        {
            saveImages = true;
        }
        while (arguments.read("--show"))
        {
            showImages = true;
        }
        while (arguments.read("--window", strWidth, strHeight))
        {
            viewWidth = std::stoi(strWidth);
            viewHeight = std::stoi(strHeight);
        }
        while (arguments.read("--log", logFile))
        {
            saveLog = true;
        }
        while (arguments.read("--logServerity", logServerity))
        {
            notifyServerity = (osg::NotifySeverity)std::stoi(logServerity);
        }
        if (pathModel.empty())
        {
            std::cout << "No path specified, please specify the path of model using the command line options below." << std::endl
                      << std::endl;
            arguments.getApplicationUsage()->write(std::cout, osg::ApplicationUsage::COMMAND_LINE_OPTION);
            return 1;
        }

        osgEarth::setNotifyLevel(notifyServerity);
        if (logFile.empty())
        {
            logFile = "log.txt";
        }
        if (saveLog)
        {
            osgEarth::setNotifyHandler(new LogFileHandler(logFile));
        }

        // ModelRenderScene("http://192.168.1.12:8010/v1/3dtiles/datasets/CgA/files/tileset.json", 800, 600);
        // ModelRenderScene renderer = ModelRenderScene("/v1/3dtiles/datasets/CgA/files/tileset.json", 800, 600);
        ModelRenderScene renderer = ModelRenderScene(pathModel, viewWidth, viewHeight);

        uint32_t count = 300, i = 0;
        // osg::Vec3d initLonLat(121.548056, 24.989444, 200);
        // osg::Vec3d initLonLat(112.990300, 28.289367, 100);
        std::array<double, 3> initLonLat = {112.9967922, 28.29139293, 101.6732864}; // 112.9967922,28.29139293,101.6732864

        // osg::Vec3d pointLonLat = initLonLat;
        std::array<double, 3> pointLonLat = initLonLat;
        // osg::Vec3d offset(0.00001, 0.00001, 0);
        std::array<double, 3> offset = {0.00001, 0.00001, 0};
        std::array<double, 3> pose = {50.81225322921506, 0.6348451336700702, 41.91234178409955};

        vector<int> compression_params;
        compression_params.push_back(cv::IMWRITE_PNG_COMPRESSION);
        compression_params.push_back(9);

        while (true)
        {
            if (i < count)
            {

                renderer.updateViewPoint(pointLonLat, pose);
                // 渲染一帧
                renderer.nextFrame();

                // 获取图像数据并处理
                if (i > 10)
                {
                    if (saveImages)
                    {
                        renderer.saveColorImage("colorImage-" + std::to_string(i) + ".png");
                        renderer.saveDepthImage("depthImage-" + std::to_string(i) + ".png");
                    }
                    // uint32_t imgHeight = renderer.getColorImageHigh();
                    // uint32_t imgWidth = renderer.getColorImageWidth();
                    cv::Mat colorImgMat = renderer.getColorImage();
                    cv::flip(colorImgMat, colorImgMat, 0);
                    cv::cvtColor(colorImgMat, colorImgMat, cv::COLOR_RGB2BGR);

                    cv::Mat depthImgMat = renderer.getDepthImage();
                    cv::flip(depthImgMat, depthImgMat, 0);

                    if (showImages)
                    {
                        std::cout << "loop number: " << i << std::endl;
                        cv::imshow("color image", colorImgMat); // 将彩色显示出来
                        cv::imshow("depth image", depthImgMat); // 将深度显示出来
                        int key = cv::waitKey(5) & 0xff;        // 捕获键值
                        if (key == 32)
                        { // 空格键实现暂停与开始
                            waitKey(0);
                        }
                    }
                    else
                    {
                        // process the images .....
                        sleep(0.04);
                    }
                }

                // 更新相机的下一个经纬度+高程位置
                pointLonLat.at(0) += offset.at(0);
                pointLonLat.at(1) += offset.at(1);
                pointLonLat.at(2) += offset.at(2);
                i++;
            }
            else
            { // reset the initial value
                pointLonLat = initLonLat;
                i = 0;
            }
        }

        // //go to the viewpoint
        // // osgEarth::Viewpoint vp;
        // // osgEarth::GeoPoint newPoint(((osgEarth::MapNode*)renderer.viewer->getSceneData())->getMap()->getSRS(), 121.548056, 24.989444, 200);
        // // vp.focalPoint() = newPoint;
        // // vp.heading()->set(0, osgEarth::Units::DEGREES);
        // // vp.pitch()->set(-60, osgEarth::Units::DEGREES);
        // // vp.range()->set(200, osgEarth::Units::METERS);
        // // ((osgEarth::EarthManipulator*)(renderer.viewer->getCameraManipulator()))->setViewpoint(vp, 5);

        // osg::Matrix mRealAttitude;
        // osg::Vec3d v3Eye, v3Center, v3Up;
        // while (!renderer.viewer->done())
        // {
        //     renderer.viewer->frame();
        //     // mRealAttitude.getLookAt(v3Eye, v3Center, v3Up);//获取新的位置和姿态

        //     // // renderer.viewer->getCamera()->getViewMatrixAsLookAt(v3Eye, v3Center, v3Up);
        //     // cout << "v3Eye.x=" << v3Eye[0] << " v3Eye.y=" << v3Eye[1] << " v3Eye.z=" << v3Eye[2] << std::endl;
        //     // cout << "v3Center.x=" << v3Center[0] << " v3Center.y=" << v3Center[1] << " v3Center.z=" << v3Center[2] << std::endl;
        //     // cout << "v3Up.x=" << v3Up[0] << " v3Up.y=" << v3Up[1] << " v3Up.z=" << v3Up[2] << std::endl;

        //     double fovy, aspectRatio, zNear, zFar;
        //     renderer.viewer->getCamera()->getProjectionMatrixAsPerspective(fovy, aspectRatio, zNear, zFar);
        //     std::cout<< "fovy: " << fovy << " " << "aspectRatio: " << aspectRatio<< " " << "zNear: "<< zNear << " " << "zFar: " <<zFar<<"\n";
        //     //capture->setFileName("Image-" + to_string(i) + ".png");
        //     // osgDB::writeImageFile(*image, "screenshot" + to_string(i) + ".png");
        //     i++;
        // }

        return 0;
    }
}
