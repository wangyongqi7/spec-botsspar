host_names=(parallel2023-02 parallel2023-03 parallel2023-04 parallel2023-05 parallel2023-06 parallel2023-07 parallel2023-08)
cd src ;
make clean && make ;
cd ..;
for host_name in ${host_names[*]} ;do
    ssh parallel@$host_name " rm -rf /home/parallel/122033910188/spec-botsspar; mkdir -p /home/parallel/122033910188/spec-botsspar"
    scp -r ./* parallel@$host_name:/home/parallel/122033910188/spec-botsspar > /dev/null
done