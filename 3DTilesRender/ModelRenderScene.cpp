#ifdef PYTHON_MODULE
#include <pybind11/numpy.h>
#include <opencv2/opencv.hpp>
#endif
#include "ModelRenderScene.h"
#include "EarthWalkManipulator.h"
#include "EGLGraphicsWindowEmbedded.h"

class SnapshotCallback : public osg::Camera::DrawCallback
{
public:
    inline SnapshotCallback(osg::ref_ptr<osg::Image> cImage)
    {
        _colorImage = cImage; // new osg::Image();
        _depthImage = new osg::Image();
    }

    inline virtual void operator()(osg::RenderInfo &renderInfo) const
    {
        osg::Camera *camera = renderInfo.getCurrentCamera();
        osg::Viewport *viewport = camera ? camera->getViewport() : 0;
        if (viewport)
        {
            // auto traits = camera->getGraphicsContext()->getTraits();
            // glReadBuffer(traits->doubleBuffer ? GL_BACK : GL_FRONT);
            //_colorImage->allocateImage(int(viewport->width()), int(viewport->height()), 1, GL_RGB, GL_UNSIGNED_BYTE);
            //_colorImage->readPixels(int(viewport->x()),int(viewport->y()),int(viewport->width()),int(viewport->height()), GL_RGBA, GL_UNSIGNED_BYTE);
            _depthImage->readImageFromCurrentTexture(renderInfo.getContextID(), true, GL_UNSIGNED_SHORT);
        }
    }

    inline osg::ref_ptr<osg::Image> getColorImage(void)
    {
        // return osg::ref_ptr<osg::Image>(reinterpret_cast<osg::Image*>(_colorImage->clone(osg::CopyOp::DEEP_COPY_ALL)));
        return _colorImage;
    }

    inline osg::ref_ptr<osg::Image> getDepthImage(void)
    {
        // return osg::ref_ptr<osg::Image>(reinterpret_cast<osg::Image*>(_depthImage->clone(osg::CopyOp::DEEP_COPY_ALL)));
        return _depthImage;
    }

    inline bool saveColorImage(const std::string fileName)
    {
        // TODO: fix save bug here!
        if (_colorImage)
        {
            // osgDB::writeImageFile(*osg::ref_ptr<osg::Image>(reinterpret_cast<osg::Image*>(_colorImage->clone(osg::CopyOp::DEEP_COPY_ALL))), fileName);
            osgDB::writeImageFile(*_colorImage, fileName);
        }
        else
        {
            return false;
        }
        return true;
    }

    inline bool saveDepthImage(const std::string fileName)
    {
        if (_depthImage)
        {
            // osgDB::writeImageFile(*osg::ref_ptr<osg::Image>(reinterpret_cast<osg::Image*>(_depthImage->clone(osg::CopyOp::DEEP_COPY_ALL))), fileName);
            osgDB::writeImageFile(*_depthImage, fileName);
        }
        else
        {
            return false;
        }
        return true;
    }

private:
    mutable osg::ref_ptr<osg::Image> _colorImage;
    mutable osg::ref_ptr<osg::Image> _depthImage;
};
osg::ref_ptr<osgViewer::Viewer> ModelRenderScene::createViewer(double fx, double fy, double cx, double cy, double imgWidth, double imgHeight)
{
    // 设置图形环境特性
    osg::ref_ptr<osg::GraphicsContext::Traits> traits = new osg::GraphicsContext::Traits();
    traits->width = _viewerWidth;
    traits->height = _viewerHeight;
    traits->windowDecoration = true;
    traits->doubleBuffer = true;
    traits->pbuffer = true;
    traits->red = 8;
    traits->blue = 8;
    traits->green = 8;
    traits->sharedContext = 0;
    traits->readDISPLAY();
    traits->setUndefinedScreenDetailsToDefaultScreen();

    // 创建 Viewer
    osg::ref_ptr<osgViewer::Viewer> viewer = new osgViewer::Viewer();
    osg::ref_ptr<osg::Camera> camera = viewer->getCamera();

    // 创建图形环境
    EGLGraphicsWindowEmbedded *eglContext = new EGLGraphicsWindowEmbedded(traits.get());
    if (eglContext->isAvailable())
        camera->setGraphicsContext(eglContext);
    else
        camera->setGraphicsContext(osg::GraphicsContext::createGraphicsContext(traits.get()));

    // 设置视口
    camera->setViewport(new osg::Viewport(0, 0, traits->width, traits->height));
    // 设置缓冲
    GLenum buffer = traits->doubleBuffer ? GL_BACK : GL_FRONT;
    camera->setDrawBuffer(buffer);
    camera->setReadBuffer(buffer);

    // 近裁剪面 & 远裁剪面，可根据需要自定义
    double zNear = 0.1;
    double zFar = 10000.0;

    // 计算 frustum 的各个参数
    // 注意：下面示例假设 fovx/fovy 为“度数”，并且投影矩阵以相机中心对称为基础
    // 若 fovx/fovy 已是弧度或需要非对称，可自行调整
    double left = -cx * zNear / fx;
    double right = (imgWidth - cx) * zNear / fx;
    double bottom = -(imgHeight - cy) * zNear / fy;
    double top = cy * zNear / fy;

    camera->setProjectionMatrix(osg::Matrix::frustum(left, right, bottom, top, zNear, zFar));
    camera->setRenderTargetImplementation(osg::Camera::FRAME_BUFFER_OBJECT);
    camera->attach(osg::Camera::COLOR_BUFFER0, _img.get());
    camera->attach(osg::Camera::DEPTH_BUFFER, (osg::Texture *)_tex);
    camera->getOrCreateStateSet()->setTextureAttributeAndModes(0, _tex, osg::StateAttribute::ON);

    // （如果你不需要操作器，可以设为NULL；如需自定义操作器，则在此添加）
    viewer->setCameraManipulator(NULL);

    // 为了获取图像的回调
    SnapshotCallback *cb = new SnapshotCallback(_img);
    camera->setFinalDrawCallback(cb);

    // 初始化视图
    viewer->realize();
    return viewer.release();
}
// osg::ref_ptr<osgViewer::Viewer> ModelRenderScene::createViewer(double fovx, double fovy)
// {
//     // 设置图形环境特性
//     osg::ref_ptr<osg::GraphicsContext::Traits> traits = new osg::GraphicsContext::Traits();
//     traits->width = _viewerWidth;
//     traits->height = _viewerHeight;
//     traits->windowDecoration = true;
//     traits->doubleBuffer = true;
//     traits->pbuffer = true;
//     traits->red = 8;
//     traits->blue = 8;
//     traits->green = 8;
//     traits->sharedContext = 0;
//     traits->readDISPLAY();
//     traits->setUndefinedScreenDetailsToDefaultScreen();

//     // 创建 Viewer
//     osg::ref_ptr<osgViewer::Viewer> viewer = new osgViewer::Viewer();
//     osg::ref_ptr<osg::Camera> camera = viewer->getCamera();

//     // 创建图形环境
//     EGLGraphicsWindowEmbedded *eglContext = new EGLGraphicsWindowEmbedded(traits.get());
//     if (eglContext->isAvailable())
//         camera->setGraphicsContext(eglContext);
//     else
//         camera->setGraphicsContext(osg::GraphicsContext::createGraphicsContext(traits.get()));

//     // 设置视口
//     camera->setViewport(new osg::Viewport(0, 0, traits->width, traits->height));
//     // 设置缓冲
//     GLenum buffer = traits->doubleBuffer ? GL_BACK : GL_FRONT;
//     camera->setDrawBuffer(buffer);
//     camera->setReadBuffer(buffer);

//     // 近裁剪面 & 远裁剪面，可根据需要自定义
//     double zNear = 0.1;
//     double zFar = 10000.0;

//     // 计算 frustum 的各个参数
//     // 注意：下面示例假设 fovx/fovy 为“度数”，并且投影矩阵以相机中心对称为基础
//     // 若 fovx/fovy 已是弧度或需要非对称，可自行调整
//     double fovxRad = osg::DegreesToRadians(fovx);
//     double fovyRad = osg::DegreesToRadians(fovy);

//     double left = -zNear * std::tan(fovxRad * 0.5);
//     double right = zNear * std::tan(fovxRad * 0.5);
//     double bottom = -zNear * std::tan(fovyRad * 0.5);
//     double top = zNear * std::tan(fovyRad * 0.5);

//     // 设置相机投影矩阵（frustum）
//     camera->setProjectionMatrix(osg::Matrix::frustum(left, right, bottom, top, zNear, zFar));
//     camera->setRenderTargetImplementation(osg::Camera::FRAME_BUFFER_OBJECT);
//     camera->attach(osg::Camera::COLOR_BUFFER0, _img.get());
//     camera->attach(osg::Camera::DEPTH_BUFFER, (osg::Texture *)_tex);
//     camera->getOrCreateStateSet()->setTextureAttributeAndModes(0, _tex, osg::StateAttribute::ON);

//     // （如果你不需要操作器，可以设为NULL；如需自定义操作器，则在此添加）
//     viewer->setCameraManipulator(NULL);

//     // 为了获取图像的回调
//     SnapshotCallback *cb = new SnapshotCallback(_img);
//     camera->setFinalDrawCallback(cb);

//     // 初始化视图
//     viewer->realize();
//     return viewer.release();
// }

// 加载3DTiles模型
osg::ref_ptr<osgEarth::MapNode> ModelRenderScene::createMapNode(void)
{
    // Map is datamodel for collection of layers.
    osg::ref_ptr<osgEarth::Map> map = new osgEarth::Map;
    // create map node
    osg::ref_ptr<osgEarth::MapNode> mapNode = new osgEarth::MapNode(map);
    osg::ref_ptr<osgEarth::Cesium::CesiumNative3DTilesLayer> modelLayer = new osgEarth::Cesium::CesiumNative3DTilesLayer();
    // Set the URL of the 3D Tiles file.
    modelLayer->setURL(_modelPath);
    modelLayer->setName("3DTiles");
    modelLayer->setMaximumScreenSpaceError(1.33);
    map->addLayer(modelLayer);
    // 关闭光照
    mapNode->getOrCreateStateSet()->setMode(GL_LIGHTING, osg::StateAttribute::OFF);

    return mapNode.release();
}

ModelRenderScene::ModelRenderScene(std::string modelPath, uint32_t viewerWidth, uint32_t viewerHeight, double fx, double fy, double cx, double cy)
{

    _modelPath = modelPath;
    _viewerWidth = viewerWidth;
    _viewerHeight = viewerHeight;

    osgEarth::initialize();
    // 创建纹理buffer
    _tex = createTexture();

    // 分配image buffer保存彩色图
    _img = new osg::Image();
    _img->allocateImage(_viewerWidth, _viewerHeight, 1, GL_RGB, GL_UNSIGNED_BYTE);

    _viewer = createViewer(fx, fy, cx, cy, _viewerWidth, _viewerHeight);
    _mapNode = createMapNode();
    _viewer->setSceneData(_mapNode);
    osgEarth::notify(osg::INFO) << "ModelRenderScene Initialized." << std::endl;
}

// 计算系数a和b

void calculateCoefficients(float zNear, float zFar, float &a, float &b)
{

    a = zFar / (zFar - zNear);

    b = zFar * zNear / (zNear - zFar);
}

// 反求z值的函数
// 反求z值的函数

float inverseZBuffer(unsigned short zBufferValue, float a, float b, int nBits)
{
    // 16位无符号整数的最大值

    const float maxZBufferValue = pow(2, nBits) - 1;

    return b / (static_cast<float>(zBufferValue) / maxZBufferValue - a);
}

cv::Mat ModelRenderScene::getColorImage(void)
{

    SnapshotCallback *cb = (SnapshotCallback *)_viewer->getCamera()->getFinalDrawCallback();
    if (!cb->getColorImage())
        return cv::Mat{};

    cv::Mat colorImgMat(_viewerHeight, _viewerWidth, CV_8UC3);
    memcpy(colorImgMat.data, cb->getColorImage()->data(), cb->getColorImage()->t() * cb->getColorImage()->s() * 3);
    return colorImgMat;
}

// 获取场景渲染图宽度
unsigned int ModelRenderScene::getColorImageWidth(void)
{
    SnapshotCallback *cb = (SnapshotCallback *)_viewer->getCamera()->getFinalDrawCallback();
    return cb->getColorImage()->s();
}

// 获取场景渲染图高度
unsigned int ModelRenderScene::getColorImageHigh(void)
{
    SnapshotCallback *cb = (SnapshotCallback *)_viewer->getCamera()->getFinalDrawCallback();
    return cb->getColorImage()->t();
}

// 获取场景深度图
// unsigned char* ModelRenderScene::getDepthImage(void)
// {
//     SnapshotCallback *cb =(SnapshotCallback *)_viewer->getCamera()->getFinalDrawCallback();
//     if (!cb->getDepthImage())
//         return NULL;

//     return cb->getDepthImage()->data();
// }

cv::Mat ModelRenderScene::getDepthImage(void)
{
    SnapshotCallback *cb = (SnapshotCallback *)_viewer->getCamera()->getFinalDrawCallback();
    if (!cb->getColorImage())
        return cv::Mat{};

    cv::Mat depthImgMat(cb->getDepthImage()->t(), cb->getDepthImage()->s(), CV_16UC1);
    try
    {
        unsigned int size = cb->getDepthImage()->getTotalDataSize();
        memcpy(depthImgMat.data, cb->getDepthImage()->data(), size);
    }
    catch (std::exception e)
    {
        osgEarth::notify(osg::FATAL) << __func__ << ": Copy the depth image failed!" << std::endl;
        cv::Mat imageFloat(depthImgMat.size(), CV_32F);
        return imageFloat;
    }

    // return depthImgMat;

    // Cheng: convert uint16 depth to float meters
    cv::flip(depthImgMat, depthImgMat, 0);
    cv::Mat imageFloat(depthImgMat.size(), CV_32F);
    int N = 16; // 16位深度缓冲
    float a, b;
    double fovy, aspectRatio, zNear = 0.0, zFar = 0.0;
    _viewer->getCamera()->getProjectionMatrixAsPerspective(fovy, aspectRatio, zNear, zFar);
    calculateCoefficients(zNear, zFar, a, b);
    // 逐像素遍历并转换
    for (int i = 0; i < depthImgMat.rows; ++i)
    {

        for (int j = 0; j < depthImgMat.cols; ++j)
        {

            unsigned short pixelValue = depthImgMat.at<unsigned short>(i, j);

            float convertedValue = inverseZBuffer(pixelValue, a, b, N);
            // std::cout << "before: " << pixelValue << " " << convertedValue << std::endl;
            imageFloat.at<float>(i, j) = convertedValue;
        }
    }

    // cv::imwrite(pathSave + "/depthImage-" + img_name + ".tiff", imageFloat);
    // std::cout << "Image conversion and saving complete." << std::endl;
    return imageFloat;
}

// 获取场景渲染图宽度
unsigned int ModelRenderScene::getDepthImageWidth(void)
{
    SnapshotCallback *cb = (SnapshotCallback *)_viewer->getCamera()->getFinalDrawCallback();
    return cb->getDepthImage()->s();
}

// 获取场景渲染图高度
unsigned int ModelRenderScene::getDepthImageHigh(void)
{
    SnapshotCallback *cb = (SnapshotCallback *)_viewer->getCamera()->getFinalDrawCallback();
    return cb->getDepthImage()->t();
}

// 保存抓取的深度图
bool ModelRenderScene::saveDepthImage(std::string fileName)
{
    // SnapshotCallback *cb = (SnapshotCallback *)_viewer->getCamera()->getFinalDrawCallback();
    // return cb->saveDepthImage(fileName);
    SnapshotCallback *cb = (SnapshotCallback *)_viewer->getCamera()->getFinalDrawCallback();
    if (!cb->getDepthImage())
        return false;

    cv::Mat depthImgMat(_viewerHeight, _viewerWidth, CV_16UC1);
    memcpy(depthImgMat.data, cb->getDepthImage()->data(), _viewerHeight * _viewerWidth * 2);
    cv::flip(depthImgMat, depthImgMat, 0);

    cv::Mat imageFloat(depthImgMat.size(), CV_32F);
    int N = 16; // 16位深度缓冲
    float a, b;
    double fovy, aspectRatio, zNear = 0.0, zFar = 0.0;

    _viewer->getCamera()->getProjectionMatrixAsPerspective(fovy, aspectRatio, zNear, zFar);
    calculateCoefficients(zNear, zFar, a, b);
    // 逐像素遍历并转换
    for (int i = 0; i < depthImgMat.rows; ++i)
    {

        for (int j = 0; j < depthImgMat.cols; ++j)
        {

            unsigned short pixelValue = depthImgMat.at<unsigned short>(i, j);

            float convertedValue = inverseZBuffer(pixelValue, a, b, N);
            // std::cout << "before: " << pixelValue << " " << convertedValue << std::endl;

            imageFloat.at<float>(i, j) = convertedValue;
        }
    }

    cv::imwrite(fileName, imageFloat);
    return true;
}

// 保存抓取的渲染图
bool ModelRenderScene::saveColorImage(std::string fileName)
{
    SnapshotCallback *cb = (SnapshotCallback *)_viewer->getCamera()->getFinalDrawCallback();
    return cb->saveColorImage(fileName);
}
// 通过经纬度，高程，旋转角度，来更新视点的位置与姿态
void ModelRenderScene::updateViewPoint(std::array<double, 3> &LonLatHigh, std::array<double, 3> &Pose)
{
    const osgEarth::SpatialReference *srs = _mapNode->getMap()->getSRS();
    // osgEarth::GeoPoint gEyeGeo(srs, LonLatHigh.x(), LonLatHigh.y(), LonLatHigh.z());
    osgEarth::GeoPoint gEyeGeo(srs, LonLatHigh.at(0), LonLatHigh.at(1), LonLatHigh.at(2));
    // 获取在WGS-84坐标系下相机的位置矩阵
    osg::Matrixd matrixWorld;
    gEyeGeo.createLocalToWorld(matrixWorld);

    // double longitude = gEyeGeo.longitude().degrees();
    // double latitude = gEyeGeo.latitude().degrees();
    // double height = gEyeGeo.getAltitude(); // 通常以米为单位

    // std::cout << "Longitude: " << longitude << " degrees" << std::endl;
    // std::cout << "Latitude: " << latitude << " degrees" << std::endl;
    // std::cout << "Height: " << height << " meters" << std::endl;
    // 相机在WGS-84坐标系下进行旋转
    matrixWorld.preMultRotate(osg::Quat(osg::DegreesToRadians(Pose.at(0)), osg::X_AXIS,   // pitch
                                        osg::DegreesToRadians(Pose.at(1)), osg::Y_AXIS,   // roll
                                        osg::DegreesToRadians(Pose.at(2)), osg::Z_AXIS)); // head

    // 计算旋转矩阵
    osg::Matrix rotate;
    rotate.setRotate(matrixWorld.getRotate());
    // 计算平移矩阵
    osg::Matrix translate;
    translate.makeIdentity();
    // 平移矩阵是相对于相机坐标系，所以要取反方向
    translate.makeTranslate(-matrixWorld.getTrans());
    // 计算旋转矩阵的逆矩阵
    osg::Matrixd inverse_cam = osg::Matrixd::inverse(rotate);
    // 逆矩阵*平移矩阵
    inverse_cam.preMult(translate);
    _viewer->getCamera()->setViewMatrix(inverse_cam);
}
// 通过旋转与平移矩阵改变视点位置与姿态
void ModelRenderScene::updateRotTransMatrix(osg::Matrixd &rotate, osg::Matrixd &translate)
{
    osg::Matrixd transformation = osg::Matrixd::inverse(rotate);
    // 逆矩阵*平移矩阵
    transformation.preMult(translate);
    _viewer->getCamera()->setViewMatrix(transformation);
}

// 通过变换矩阵改变视点位置与姿态
void ModelRenderScene::updateViewMatrix(osg::Matrixd &transformation)
{
    _viewer->getCamera()->setViewMatrix(transformation);
}

// 创建纹理
osg::ref_ptr<osg::Texture2D> ModelRenderScene::createTexture(void)
{
    osg::ref_ptr<osg::Texture2D> texture = new osg::Texture2D;
    texture->setSourceFormat(GL_DEPTH_COMPONENT);
    texture->setSourceType(GL_UNSIGNED_SHORT);
    texture->setInternalFormat(GL_DEPTH_COMPONENT16);
    texture->setTextureSize(_viewerWidth, _viewerHeight);

    texture->setFilter(osg::Texture::MIN_FILTER, osg::Texture::LINEAR);
    texture->setFilter(osg::Texture::MAG_FILTER, osg::Texture::LINEAR);
    texture->setWrap(osg::Texture2D::WRAP_S, osg::Texture2D::CLAMP_TO_EDGE);
    texture->setWrap(osg::Texture2D::WRAP_T, osg::Texture2D::CLAMP_TO_EDGE);
    return texture.release();
}

osg::ref_ptr<osg::Geode> ModelRenderScene::createQuad(osg::Texture2D *texture)
{
    osg::ref_ptr<osg::Geometry> quadGeometry = osg::createTexturedQuadGeometry(osg::Vec3(0, 0, 0), osg::Vec3(_viewerWidth / 4, 0, 0), osg::Vec3(0, _viewerHeight / 4, 0));
    const std::string vertShader = ""
                                   "void main()		"
                                   "{			"
                                   "    gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;"
                                   "    gl_TexCoord[0] = gl_TextureMatrix[0] *gl_MultiTexCoord0;"
                                   "}";

    const std::string fragShader = ""
                                   "uniform sampler2D texture0;"
                                   "void main()		"
                                   "{			"
                                   "    float d = texture2D( texture0, gl_TexCoord[0].xy ).x;"
                                   "    gl_FragColor = vec4(d, d, d, 1);			"
                                   "}";

    osg::ref_ptr<osg::Shader> vert = new osg::Shader(osg::Shader::VERTEX, vertShader);
    osg::ref_ptr<osg::Shader> frag = new osg::Shader(osg::Shader::FRAGMENT, fragShader);

    osg::ref_ptr<osg::Program> program = new osg::Program;
    program->addShader(vert);
    program->addShader(frag);

    osg::ref_ptr<osg::StateSet> stateSet = quadGeometry->getOrCreateStateSet();
    stateSet->setAttributeAndModes(program);
    stateSet->addUniform(new osg::Uniform(osg::Uniform::SAMPLER_2D, "texture0", 0));
    // Apply texture to quad
    stateSet->setTextureAttributeAndModes(0, texture, osg::StateAttribute::ON);
    // PUT PLANE INTO NODE
    osg::ref_ptr<osg::Geode> quad = new osg::Geode;
    quad->addDrawable(quadGeometry);
    // DISABLE SHADOW / LIGHTNING EFFECTS
    int values = osg::StateAttribute::OFF | osg::StateAttribute::PROTECTED;
    // quad->getOrCreateStateSet()->setAttribute(new osg::PolygonMode(osg::PolygonMode::FRONT_AND_BACK, osg::PolygonMode::FILL), values);
    quad->getOrCreateStateSet()->setMode(GL_LIGHTING, values);
    quad->getOrCreateStateSet()->setTextureAttribute(0, texture);
    return quad.release();
}

void ModelRenderScene::nextFrame()
{
    OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_mutex);
    // if(_frame.valid() && !viewer->done())
    if (!_viewer->done())
    {
        //_viewer->updateTraversal();
        double fovy, ar, zNear, zFar;

        double aspectRatio = double(_viewerWidth) / double(_viewerHeight);
        _viewer->getCamera()->getProjectionMatrixAsPerspective(fovy, ar, zNear, zFar);
        // _viewer->getCamera()->setProjectionMatrixAsPerspective(_fovy, aspectRatio, zNear, zFar);
        // _viewer->getCamera()->getProjectionMatrixAsPerspective(fovy, ar, zNear, zFar);

        _viewer->frame();
        // osgEarth::notify(osg::INFO) << "fovy: " << fovy << " "
        //                             << "aspectRatio: " << aspectRatio << " "
        //                             << "zNear: " << zNear << " "
        //                             << "zFar: " << zFar << std::endl;
    }
    else
    {
        osgEarth::notify(osg::FATAL) << "Viewer or Camera invalid.";
    }
}

ModelRenderScene::~ModelRenderScene()
{
    // TODO Auto-generated destructor stub
    _viewer->setDone(true);
    _viewer->setReleaseContextAtEndOfFrameHint(true);

#ifdef DEBUG
    std::cout << "ModelRenderScene deleted." << std::endl;
#endif
}