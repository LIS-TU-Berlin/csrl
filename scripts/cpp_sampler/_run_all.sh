./x.exe -scene double_sphere -num_samples 10000 & 
sleep 2
./x.exe -scene sphere_cube -num_samples 20000 &
sleep 2
./x.exe -scene panda_sphere -num_samples 10000 & 
sleep 2
./x.exe -scene panda_cube -num_samples 20000 &
sleep 2
./x.exe -scene allegro_cube -num_samples 50000 &
sleep 2
./x.exe -scene allegroDown_cube -num_samples 50000 &
sleep 2

#wait
