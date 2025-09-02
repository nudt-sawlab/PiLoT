#include <osgEarth/Notify>
#include <fstream>

class LogFileHandler : public osg::NotifyHandler
{
public:
    LogFileHandler( const std::string& file ) 
    { 
        _log.open( file.c_str(), std::ios::out); 
    }
    virtual ~LogFileHandler() { _log.close(); }
    virtual void notify(osg::NotifySeverity severity, const char* msg)
    { 
        std::string logStr(msg);
        size_t pos = logStr.rfind('\n');
        // 如果找到了换行符
        if (pos != std::string::npos) {
            // 在换行符的位置删除一个字符，即删除换行符
            logStr.erase(pos, 1);
        }
        _log << logStr << std::endl;
    }

protected:
    std::ofstream _log;
};

