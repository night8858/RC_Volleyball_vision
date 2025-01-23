#include "my_serial.h"

int main()
{
    std::cout << "hello" << std::endl;
    toe::serial_port templ;
    templ.read_port();
    std::vector<float> se = {-1,-1,-1,100,100,100,100,0};
    templ.write_port(se);
}