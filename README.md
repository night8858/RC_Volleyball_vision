# 此项目是基于nvidia的排球识别项目，还在建设中

# 依赖安装
## intel realsence
RealSense的SDK2.0安装

1.注册服务器的公钥
```
sudo apt-get update && sudo apt-get upgrade && sudo apt-get dist-upgrade
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE || sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE

```

2.将服务器添加到存储库列表中

```
sudo add-apt-repository "deb https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main" -u

```

3安装SDK2
```
sudo apt-get install librealsense2-dkms
sudo apt-get install librealsense2-utils

```

选装
```
sudo apt-get install librealsense2-dev
sudo apt-get install librealsense2-dbg

```

连接D435i测试
```
realsense-viewer 

```

## opencv

## tensorRT

## cuda