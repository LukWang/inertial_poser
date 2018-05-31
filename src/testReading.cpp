#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

int main()
{
    ifstream in("/home/luk/Public/Total Capture/S3/acting1_BlenderZXY_YmZ.bvh");
    int joint_count = 0;
    while(1)
    {
        char line[256];
        string::size_type idx;
        string motion = "MOTION";
        string channel = "CHANNELS";
        if(in.getline(line, 256).good())
        {
            string str(line);
            idx = str.find(channel);
            if(idx != string::npos)
            {
                joint_count++;
            }
            idx = str.find(motion);
            if(idx != string::npos)
            {
                in.getline(line, 256);
                in.getline(line, 256);
                break;
            }
        }
    }

    cout << "joint count: " << joint_count << endl;
    double data;
    vector<double> joints;
    for (int i = 0; i < joint_count * 6; i++)
    {
        int joint_index = i/6;
        int data_index = i%6;
        in >> data;
        if(!(data_index < 3))
        if(!(joint_index == 5 || joint_index == 6 || joint_index == 10 || joint_index == 11 || joint_index == 12 || joint_index >=16))
        {
            joints.push_back(data);
            cout << data << endl;
        }
    }

    cout << "joint_num: " << joints.size() << endl;
    in >> data;
    cout << "line 2:" << data << endl;

    return 0;
}
