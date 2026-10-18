# 开启最高级别优化 (-O3)、硬件本地指令集优化 (-march=native)、OpenMP 多线程支持以及生成位置无关代码 (-fPIC)
gcc -O3 -march=native -fopenmp -fPIC -shared -o liblinear.so linear.c -lm


gcc -O3 -march=native linear.c test_linear.c -o test_linear -lm
./test_linear