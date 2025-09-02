#ifndef MODEL_RENDER_SCENE_H
#define MODEL_RENDER_SCENE_H

#include <osg/ref_ptr>
#include <osg/Array>
#include <osg/ImageUtils>
#include <osgGA/StateSetManipulator>
#include <osgViewer/Viewer>
#include <osg/GraphicsContext>
#include <osg/Texture2D>
#include <osg/FrameBufferObject>
#include <osgDB/WriteFile>
#include <osg/Referenced>
#include <osg/Vec3>
#include <osg/Image>
#include <osg/State>

#include <osgEarth/Metrics>
#include <osgEarth/GDAL>
#include <osgEarth/GeoTransform>
#include <osgEarth/MapNode>
#include <osgEarth/ShaderGenerator>
#include <osgEarth/Viewpoint>
#include <osgEarth/GeoTransform>
#include <osgEarth/ModelLayer>
#include <osgEarthCesium/CesiumLayer>
#include <osgEarth/EarthManipulator>
#include <osgEarth/Notify>

#include <string>
#include <chrono>
#include <thread>
#include <assert.h>
#include <opencv2/opencv.hpp>

#include <EGL/egl.h>

static const EGLint configAttribs[] = {
    EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
    EGL_BLUE_SIZE, 8,
    EGL_GREEN_SIZE, 8,
    EGL_RED_SIZE, 8,
    EGL_DEPTH_SIZE, 8,
    EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
    EGL_NONE};

static const int pbufferWidth = 9;
static const int pbufferHeight = 9;

static const EGLint pbufferAttribs[] = {
    EGL_WIDTH,
    pbufferWidth,
    EGL_HEIGHT,
    pbufferHeight,
    EGL_NONE,
};

class ModelRenderScene
{
protected:
    EGLDisplay eglDpy;
    EGLint major, minor;
    EGLint numConfigs;
    EGLConfig eglCfg;
    EGLSurface eglSurf;
    EGLContext eglCtx;

public:
    ModelRenderScene(std::string modelPath, uint32_t viewerWidth, uint32_t viewerHeight, double fx, double fy, double cx, double cy);
    virtual ~ModelRenderScene();
    // 渲染下一帧
    void nextFrame();
    // 通过更新视点（相机）的经纬度、高、姿态，来改变视点（相机）位置与姿态
    // void updateViewPoint(osg::Vec3d &LonLatHigh, osg::Vec3d &Pose);
    void updateViewPoint(std::array<double, 3> &LonLatHigh, std::array<double, 3> &Pose);
    inline OpenThreads::Mutex *GetMutexObject(void) { return &_mutex; }
    void updateViewMatrix(osg::Matrixd &transformation);
    // 通过更新视点（相机）的旋转和平移矩阵，来改变视点（相机）位置与姿态
    void updateRotTransMatrix(osg::Matrixd &rotate, osg::Matrixd &translate);
    // 获取场景渲染图
    cv::Mat getColorImage(void);
    // 获取场景渲染图宽度
    unsigned int getColorImageWidth(void);
    unsigned int getColorImageHigh(void);
    // 获取场景深度图
    cv::Mat getDepthImage(void);
    // 获取场景深度图宽度
    unsigned int getDepthImageWidth(void);
    // 获取场景深度图高度
    unsigned int getDepthImageHigh(void);
    // 保存抓取的深度图
    bool saveDepthImage(std::string fileName);
    // 保存抓取的渲染图
    bool saveColorImage(std::string fileName);

private:
    osg::ref_ptr<osg::Texture2D> createTexture(void);
    osg::ref_ptr<osgEarth::MapNode> createMapNode(void);
    osg::ref_ptr<osgViewer::Viewer> createViewer(double fx, double fy, double cx, double cy, double imgWidth, double imgHeight);
    osg::ref_ptr<osg::Geode> createQuad(osg::Texture2D *texture);

private:
    osg::ref_ptr<osgViewer::Viewer> _viewer;
    std::string _modelPath;
    uint32_t _viewerWidth;
    uint32_t _viewerHeight;
    osg::ref_ptr<osgEarth::MapNode> _mapNode;
    osg::ref_ptr<osg::Texture2D> _tex;
    mutable osg::ref_ptr<osg::Image> _img;
    mutable OpenThreads::Mutex _mutex;
};

#endif