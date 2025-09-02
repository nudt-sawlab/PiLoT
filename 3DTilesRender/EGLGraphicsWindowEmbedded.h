#ifndef EGL_GRAPHICS_WINDOW_EMBEDDED_H
#define EGL_GRAPHICS_WINDOW_EMBEDDED_H

#include <osgDB/ReadFile>
#include <osgUtil/Optimizer>
#include <osg/CoordinateSystemNode>

#include <osg/Switch>
#include <osg/Types>
#include <osgText/Text>

#include <osgViewer/Viewer>
#include <osgViewer/ViewerEventHandlers>

#include <osgGA/TrackballManipulator>
#include <osgGA/FlightManipulator>
#include <osgGA/DriveManipulator>
#include <osgGA/KeySwitchMatrixManipulator>
#include <osgGA/StateSetManipulator>
#include <osgGA/AnimationPathManipulator>
#include <osgGA/TerrainManipulator>
#include <osgGA/SphericalManipulator>

#include <osgGA/Device>

#include <iostream>

#include <osg/LineWidth>
#include <osg/Point>
#include <osg/MatrixTransform>
#include <osg/io_utils>

#define EGL_EGLEXT_PROTOTYPES
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <cstdio>

class EGLGraphicsWindowEmbedded : public osgViewer::GraphicsWindowEmbedded
{
protected:
    EGLDisplay  eglDpy;
    EGLint      major, minor;
    EGLint      numConfigs;
    EGLConfig   eglCfg;
    EGLSurface  eglSurf;
    EGLContext  eglCtx;

    bool available;
public:
    bool isAvailable() {
        return available;
    }

    EGLGraphicsWindowEmbedded(osg::GraphicsContext::Traits* traits=0) //: osgViewer::GraphicsWindowEmbedded(traits)
    {
        const EGLint configAttribs[] = {
              EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
              EGL_BLUE_SIZE, 8,
              EGL_GREEN_SIZE, 8,
              EGL_RED_SIZE, 8,
              EGL_DEPTH_SIZE, 8,
              EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
              EGL_NONE
        };

        static const int pbufferWidth = 256;
        static const int pbufferHeight = 256;

        const EGLint pbufferAttribs[] = {
            EGL_WIDTH, pbufferWidth,
            EGL_HEIGHT, pbufferHeight,
            EGL_NONE,
        };

        static const int MAX_DEVICES = 16;
        EGLDeviceEXT eglDevs[MAX_DEVICES];
        EGLint numDevices;

        PFNEGLQUERYDEVICESEXTPROC eglQueryDevicesEXT =
            (PFNEGLQUERYDEVICESEXTPROC)
            eglGetProcAddress("eglQueryDevicesEXT");

        if (eglQueryDevicesEXT == NULL) {
            available = false;
            return;
        } else
            available = true;

        eglQueryDevicesEXT(MAX_DEVICES, eglDevs, &numDevices);
        // Nvidia laptop: 3 Devices, DEFAULT x-server, Intel Mesa, NVIDIA GPU.
        // We should select the NVIDIA GPU to support off-screen render with EGL.
        printf("EGL Detected %d devices\n", numDevices); 

        if (numDevices == 0) {
            available = false;
            return;
        } else
            available = true;

        PFNEGLGETPLATFORMDISPLAYEXTPROC eglGetPlatformDisplayEXT =
            (PFNEGLGETPLATFORMDISPLAYEXTPROC)
            eglGetProcAddress("eglGetPlatformDisplayEXT");

        // 仅有NVIDIA card支持离屏渲染，如果numDevices < 3，则没有独显。
        int sel = 0;
        if (numDevices >= 3)
            sel = 2;
        else if (numDevices == 2)
            sel = 1;
        eglDpy = eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT, 
                                            eglDevs[sel], 0);

        available = false;
        for(int i = 0; i < numDevices; ++i) {
            eglDpy = eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT, 
                                            eglDevs[i], 0);
            printf("DEBUG: eglDpy %d, %lld\n", i, eglDpy);
            if (eglDpy != EGL_NO_DISPLAY && eglInitialize(eglDpy, &major, &minor)) {
                 // 2. Select an appropriate configuration
                eglChooseConfig(eglDpy, configAttribs, &eglCfg, 1, &numConfigs);

                // 3. Create a surface
                eglSurf = eglCreatePbufferSurface(eglDpy, eglCfg, pbufferAttribs);
                if (eglSurf) {
                    available = true;
                    break;
                }
            }
        }

        if (!available)
            return;

        // 1. Initialize EGL
        // eglDpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);
        eglInitialize(eglDpy, &major, &minor);

        // 2. Select an appropriate configuration
        eglChooseConfig(eglDpy, configAttribs, &eglCfg, 1, &numConfigs);

        // 3. Create a surface
        eglSurf = eglCreatePbufferSurface(eglDpy, eglCfg, pbufferAttribs);

        // 4. Bind the API
        eglBindAPI(EGL_OPENGL_API);

        // 5. Create a context and make it current
        eglCtx = eglCreateContext(eglDpy, eglCfg, EGL_NO_CONTEXT, NULL);

        init();

    }
    virtual ~EGLGraphicsWindowEmbedded()
    {
        // 6. Terminate EGL when finished
        eglTerminate(eglDpy);
    }

    virtual bool isSameKindAs(const Object* object) const { return dynamic_cast<const EGLGraphicsWindowEmbedded*>(object)!=0; }
    virtual const char* libraryName() const { return "osgViewer"; }
    virtual const char* className() const { return "EGLGraphicsWindowEmbedded"; }

        void init()
        {
            if (valid())
            {
                setState( new osg::State );
                getState()->setGraphicsContext(this);

                getState()->setContextID( osg::GraphicsContext::createNewContextID() );
            }
        }

        // dummy implementations, assume that graphics context is *always* current and valid.
        virtual bool valid() const { return true; }
        virtual bool realizeImplementation() { return true; }
        virtual bool isRealizedImplementation() const  { return true; }
        virtual void closeImplementation() {}
        virtual bool makeCurrentImplementation() {
            eglMakeCurrent(eglDpy, eglSurf, eglSurf, eglCtx);
            return true;
        }
        virtual bool releaseContextImplementation() { return true; }
        virtual void swapBuffersImplementation() {}
        virtual void grabFocus() {}
        virtual void grabFocusIfPointerInWindow() {}
        virtual void raiseWindow() {}
};

#endif