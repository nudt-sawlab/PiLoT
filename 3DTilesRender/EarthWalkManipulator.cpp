#include "EarthWalkManipulator.h"
#include <osgViewer/Viewer>

#include <osgDB/ReadFile>
#include <osg/MatrixTransform>


using namespace osgEarth;

inline void dumpMatrix4D(osg::Matrixd &matrix)
{
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            std::cout << std::fixed << std::setw(8) << std::setfill('0') << matrix(i,j) << "\t";
        }
        std::cout << std::endl; // 每行结束换行
    }
}

EarthWalkManipulator::EarthWalkManipulator()
{
    //_eye = osg::Vec3d(0, 0, 0);
    osg::Matrix matrixGood1;
    GeoPoint gEyeGeo(_srs, 121.548056, 24.989444, 50);
    gEyeGeo.createLocalToWorld(matrixGood1);
    _eye = matrixGood1.getTrans();
    //_rotate = osg::Quat(-osg::PI_2, osg::X_AXIS);
    _speed = 1.0;
}

EarthWalkManipulator::~EarthWalkManipulator()
{
}

//获取相机的姿态矩阵 
osg::Matrixd EarthWalkManipulator::getMatrix() const
{
    osg::Matrix mat;
    
    mat.setRotate(_rotate);//先旋转
    std::cout << "Camera Pose in World coordinate: w=" << _rotate[0] << " x=" << _rotate[1] << " y=" << _rotate[2] << " z=" << _rotate[3] << std::endl;
    mat.postMultTranslate(_eye);//再平移
    std::cout << "Camera Position in World coordinate: x=" << _eye[0] << " y=" << _eye[1] << " z=" << _eye[2] << std::endl;
    return mat;
}

osg::Matrixd EarthWalkManipulator::getInverseMatrix() const
{
    osg::Matrix mat;
    
    mat.setRotate(-_rotate);
    std::cout << "Camera Pose in view coordinate: w=" << _rotate[0] << " x=" << _rotate[1] << " y=" << _rotate[2] << " z=" << _rotate[3] << std::endl;
    mat.preMultTranslate(-_eye);
    std::cout << "Camera Position in view coordinate: x=" << -_eye[0] << " y=" << -_eye[1] << " z=" << -_eye[2] << std::endl;
    return mat;
    //return osg::Matrixd::inverse(getMatrix());
}

void EarthWalkManipulator::home(double unused)
{
    //_eye = osg::Vec3d(0, 0, 0);
    osg::Matrix matrixGood1;
    GeoPoint gEyeGeo(_srs, 121.548056, 24.989444, 200);
    gEyeGeo.createLocalToWorld(matrixGood1);
    _eye = matrixGood1.getTrans();
    _speed = 1.0;
}

void
EarthWalkManipulator::home(const osgGA::GUIEventAdapter&, osgGA::GUIActionAdapter& us)
{
    home(0.0);
    us.requestRedraw();
}

void EarthWalkManipulator::setNode(osg::Node* node)
{
    // you can only set the node if it has not already been set, OR if you are setting
    // it to NULL. (So to change it, you must first set it to NULL.) This is to prevent
    // OSG from overwriting the node after you have already set on manually.
    if (node == 0L || !_node.valid())
    {
        _root = node;
        _node = node;
        _mapNode = 0L;
        _srs = 0L;

        established();

        osg::Matrix matrixGood1;
        //GeoPoint point1(_srs, 0, 0, 10000.0);
        GeoPoint point1(_srs, 121.548056, 24.989444, 200);
        point1.createLocalToWorld(matrixGood1);

        _eye = matrixGood1.getTrans();

        osg::Vec3d worldup;
        point1.createWorldUpVector(worldup);

        osg::Matrix mat;
        matrixGood1.getRotate().get(mat);
        dumpMatrix4D(mat);

        osg::Vec3d eye, center, up;
        mat.getLookAt(eye, center, up);
        std::cout << "eye Position in view coordinate: x=" << eye[0] << " y=" << eye[1] << " z=" << eye[2] << std::endl;
        std::cout << "center Position in view coordinate: x=" << center[0] << " y=" << center[1] << " z=" << center[2] << std::endl;
        std::cout << "up Position in view coordinate: x=" << up[0] << " y=" << up[1] << " z=" << up[2] << std::endl;
        mat.makeLookAt(_eye, -worldup, up);

        _rotate = mat.getRotate();

    }
}

osg::Node* EarthWalkManipulator::getNode()
{
    return _node.get();
}

bool EarthWalkManipulator::handle(const osgGA::GUIEventAdapter& ea, osgGA::GUIActionAdapter& us)
{
    switch (ea.getEventType())
    {
        case(osgGA::GUIEventAdapter::FRAME):
        {
            if (calcMovement(ea))//根据鼠标在屏幕中的位置调整相机转向
                us.requestRedraw();
            return true;
        }
        break;
        case(osgGA::GUIEventAdapter::SCROLL):
        {
            osg::Quat qat;
            osg::Matrix mat;
            _rotate.get(mat);
            osg::Vec3d eye, center, up;
            mat.getLookAt(eye, center, up);

            osg::Vec3d dirction = center - eye;
            dirction.normalize();
            up.normalize();
            osg::Vec3d cross = dirction^up;
            cross.normalize();
            cross *= 0.01;
            switch (ea.getScrollingMotion())
            {
                case osgGA::GUIEventAdapter::ScrollingMotion::SCROLL_UP://逆时针旋转相机
                {
                    mat = osg::Matrix::lookAt(eye, center, up + cross);
                    _rotate = mat.getRotate();
                }
                break;
                case osgGA::GUIEventAdapter::ScrollingMotion::SCROLL_DOWN://顺时针旋转相机
                {
                    mat = osg::Matrix::lookAt(eye, center, up - cross);
                    _rotate = mat.getRotate();
                }
                break;
            }
            return true;
        }
        break;
        case (osgGA::GUIEventAdapter::KEYDOWN):
        {
            osg::Vec3   v3Direction;         //视点方向  
            osg::Matrix mCameraQuat;
            osg::Vec3d  v3Eye, v3Center, v3Up;
            _rotate.get(mCameraQuat);
            mCameraQuat.getLookAt(v3Eye, v3Center, v3Up);//这里的v3Eye不是实际相机的位置，而是0，0，0
            v3Direction = v3Center - v3Eye;
            v3Direction.normalize();
            osg::Vec3d v3CrossVector = v3Up^v3Direction;
            v3CrossVector.normalize();
            if (ea.getKey() == 'w' || ea.getKey() == 'W' || ea.getKey() == osgGA::GUIEventAdapter::KEY_Up)//前进
            {
                _eye += v3Direction * _speed;
            }
            if (ea.getKey() == 's' || ea.getKey() == 'S' || ea.getKey() == osgGA::GUIEventAdapter::KEY_Down)//后退
            {
                _eye -= v3Direction * _speed;
            }
            if (ea.getKey() == 'a' || ea.getKey() == 'A' || ea.getKey() == osgGA::GUIEventAdapter::KEY_Left)//左移
            {
                _eye += v3CrossVector * _speed;
            }
            if (ea.getKey() == 'd' || ea.getKey() == 'D' || ea.getKey() == osgGA::GUIEventAdapter::KEY_Right)//右移
            {
                _eye -= v3CrossVector * _speed;
            }
            if (ea.getKey() == '-' || ea.getKey() == '_' || ea.getKey() == osgGA::GUIEventAdapter::KEY_Control_L)//减10倍移动速度
            {
                _speed /= 10.0;
                if (_speed < 1.0)
                {
                    _speed = 1.0;
                }
            }
            if (ea.getKey() == '=' || ea.getKey() == '+' || ea.getKey() == osgGA::GUIEventAdapter::KEY_Shift_L)//加10倍移动速度
            {
                _speed *= 10.0;
                if (_speed > 100000.0)
                {
                    _speed = 100000.0;
                }
            }

            if (ea.getKey() == 'h' || ea.getKey() == 'H')//在当前经纬度，姿态回正：1.视点向地面 2.头部向正北
            {
                v3Eye = _eye;//使用相机实际位置
                osg::Vec3d v3EyeLonLat;
                _srs->transformFromWorld(v3Eye, v3EyeLonLat);
                //先获取当前位置的经纬度，再获取当前正上，正北
                osg::Matrix mRealAttitude;

                if (v3EyeLonLat.z() < 0)//v3EyeLonLat.z()是眼点实际海拔
                    v3EyeLonLat.z() = 100;//将海拔0以下的物体拉到海拔100米

                GeoPoint gEyeGeo(_srs, v3EyeLonLat.x(), v3EyeLonLat.y(), v3EyeLonLat.z());
                gEyeGeo.createLocalToWorld(mRealAttitude);

                osg::Vec3d v3HorizonUp;//指天向量
                gEyeGeo.createWorldUpVector(v3HorizonUp);
                
                _eye = mRealAttitude.getTrans();

                mRealAttitude.getLookAt(v3Eye, v3Center, v3Up);//获取新的位置和姿态

                osg::Matrix mDeviationAttitude;//向北位置偏移0.00001纬度，为了计算正北方向
                GeoPoint gDeviationEyeGeo(_srs, v3EyeLonLat.x(), v3EyeLonLat.y() + 0.00001, v3EyeLonLat.z());
                gDeviationEyeGeo.createLocalToWorld(mDeviationAttitude);
                osg::Vec3d v3DeviationNorthPoint = mDeviationAttitude.getTrans();
                osg::Vec3d v3NorthHeadUp = v3DeviationNorthPoint - v3Eye;
                v3NorthHeadUp.normalize();//指北向量

                //if (v3EyeLonLat.y() < 90.0 && v3EyeLonLat.y() > 0.0)//没研究出为什么北半球和南半球需要相反，但实际使用没问题
                //{
                //    mRealAttitude.makeLookAt(osg::Vec3d(0,0,0), -v3HorizonUp, -v3NorthHeadUp);
                //}
                if (v3EyeLonLat.y() < 89.99999  && v3EyeLonLat.y() > -90.0)
                {
                    mRealAttitude.makeLookAt(osg::Vec3d(0, 0, 0), -v3HorizonUp, v3NorthHeadUp);
                }
                _rotate = mRealAttitude.getRotate();
            }
        }break;
        default:
            return false;
    }
}

bool EarthWalkManipulator::established()
{
    if (_srs.valid() && _mapNode.valid() && _node.valid())
        return true;

    // lock down the observed node:
    osg::ref_ptr<osg::Node> safeNode;
    if (!_node.lock(safeNode))
        return false;

    // find a map node or fail:
    _mapNode = osgEarth::MapNode::findMapNode(safeNode.get());
    if (!_mapNode.valid())
        return false;

    // Cache the SRS.
    _srs = _mapNode->getMapSRS();
    return true;
}

void EarthWalkManipulator::addMouseEvent(const osgGA::GUIEventAdapter& ea)
{
    _ga_t1 = _ga_t0;
    _ga_t0 = &ea;
}

bool EarthWalkManipulator::calcMovement(const osgGA::GUIEventAdapter& ea)
{
    osg::Quat qat;
    osg::Matrix mat;
    _rotate.get(mat);
    osg::Vec3d eye, center, up;
    mat.getLookAt(eye, center, up);

    osg::Vec3d dirction = center - eye;
    dirction.normalize();
    up.normalize();
    osg::Vec3d cross = dirction^up;
    cross.normalize();

    double x1 = ea.getXnormalized();
    double y1 = ea.getYnormalized();

    osg::Vec3d deviation(0, 0, 0);
    if (x1 > 0.1)
    {
        deviation += cross * 0.001;
    }
    else if (x1 < -0.1)
    {
        deviation -= cross * 0.001;
    }
    if (y1 > 0.1)
    {
        deviation += up * 0.001;
    }
    else if (y1 < -0.1)
    {
        deviation -= up * 0.001;
    }

    mat = osg::Matrix::lookAt(eye, deviation + center, up);
    _rotate = mat.getRotate();

    return true;
}