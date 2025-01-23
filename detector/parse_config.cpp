#include <jsoncpp/json/json.h>
#include <string>
#include <fstream>
#include <iostream>
#include "common_struct.hpp"

using namespace std;


void load_config(Appconfig& config, std::string json_file_path)
{
    Json::Reader reader;
    Json::Value value;
    ifstream in(json_file_path, ios::binary);
    cout << "load json now..." << endl;
    if (!in.is_open())
    {
        cerr << "Failed to open file: " << json_file_path;
        exit(1);
    }
    if (reader.parse(in, value))
    {


        config.detect_config.engine_file_path = value["path"]["engine_file_path"].asString();

        config.detect_config.batch_size = value["NCHW"]["batch_size"].asInt();
        config.detect_config.c = value["NCHW"]["C"].asInt();
        config.detect_config.w = value["NCHW"]["W"].asInt();
        config.detect_config.h = value["NCHW"]["H"].asInt();

        config.detect_config.type = value["img"]["type"].asInt();
        config.detect_config.width = value["img"]["width"].asInt();
        config.detect_config.height = value["img"]["height"].asInt();

        config.detect_config.nms_thresh = value["thresh"]["nms_thresh"].asFloat();
        config.detect_config.bbox_conf_thresh = value["thresh"]["bbox_conf_thresh"].asFloat();
        config.detect_config.merge_thresh = value["thresh"]["merge_thresh"].asFloat();

        config.detect_config.classes = value["nums"]["classes"].asInt();
        config.detect_config.sizes = value["nums"]["sizes"].asInt();
        config.detect_config.colors = value["nums"]["colors"].asInt();
        config.detect_config.kpts = value["nums"]["kpts"].asInt();

        for (auto i=0; i < value["anchors"]["1"].size(); i++)
            config.detect_config.a1.emplace_back(value["anchors"]["1"][i].asFloat());
        for (auto i=0; i < value["anchors"]["2"].size(); i++)
            config.detect_config.a2.emplace_back(value["anchors"]["2"][i].asFloat());
        for (auto i=0; i < value["anchors"]["3"].size(); i++)
            config.detect_config.a3.emplace_back(value["anchors"]["3"][i].asFloat());
        for (auto i=0; i < value["anchors"]["4"].size(); i++)
            config.detect_config.a4.emplace_back(value["anchors"]["4"][i].asFloat());

        config.camera_config.device_id = value["camera_0"]["device_id"].asInt();
        config.camera_config.exposure = value["camera_0"]["exposure"].asInt();
        config.camera_config.height = value["camera_0"]["height"].asInt();
        config.camera_config.width = value["camera_0"]["width"].asInt();
        config.camera_config.offset_x = value["camera_0"]["offset_x"].asInt();
        config.camera_config.offset_y = value["camera_0"]["offset_y"].asInt();
        config.detect_config.z_scale = value["z_scale"].asFloat();
        config.detect_config.z_scale_right = value["z_scale_right"].asFloat();
    }
    else
    {
        cerr << "Load Json Error!!!" << endl;
        exit(1);
    }
    cout << "load json success" << endl;
}

