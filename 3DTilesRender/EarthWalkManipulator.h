#ifndef __EARTH_WALK_MANIPULATOR__
#define __EARTH_WALK_MANIPULATOR__
#include <osgGA/CameraManipulator>
#include <osgEarth/MapNode>
#include <osgEarth/Viewpoint>
#include <osgEarth/GeoData>

class EarthWalkManipulator :public osgGA::CameraManipulator
{
public:
    EarthWalkManipulator();
    ~EarthWalkManipulator();

    //所有漫游器都必须实现的4个纯虚函数  
    virtual void setByMatrix(const osg::Matrixd& matrix) {}  //设置相机的位置姿态矩阵  
    virtual void setByInverseMatrix(const osg::Matrixd& matrix) {}  //设置相机的视图矩阵  
    virtual osg::Matrixd getMatrix() const;  //获取相机的姿态矩阵 
    virtual osg::Matrixd getInverseMatrix() const;   //获取相机的视图矩阵 

    //所有操作在这里响应
    virtual bool handle(const osgGA::GUIEventAdapter& ea, osgGA::GUIActionAdapter& us);

    // Attach a node to the manipulator.
    virtual void setNode(osg::Node*);
    virtual osg::Node* getNode();
    bool established();

    //virtual void computeHomePosition();

    /**
    * Sets the camera position, optionally moving it there over time.
    */
    //virtual void setViewpoint(const osgEarth::Viewpoint& vp, double duration_s = 0.0);
    virtual void home(double /*unused*/);
    virtual void home(const osgGA::GUIEventAdapter& ea, osgGA::GUIActionAdapter& us);

    void addMouseEvent(const osgGA::GUIEventAdapter& ea);
    bool calcMovement(const osgGA::GUIEventAdapter& ea); 
    
    void flushMouseEventStack();
protected:
    osg::Vec3   _eye;               //视点位置  
    osg::Quat    _rotate;            //旋转姿态 
    osg::ref_ptr<osg::Node>  _root;

    osg::observer_ptr<osg::Node> _node;
    osg::observer_ptr<osgEarth::MapNode>   _mapNode;

    osg::ref_ptr<const osgEarth::SpatialReference> _srs;

    float        _speed;                //速度
    bool        _transversal;        //横移
    // Internal event stack comprising last two mouse events.
    osg::ref_ptr<const osgGA::GUIEventAdapter> _ga_t1;
    osg::ref_ptr<const osgGA::GUIEventAdapter> _ga_t0;
};
#endif