# bash transfer.sh
echo "*****************************************************************************************************************************************************"
m=50
n=100
t=10
np=7
echo "begin mpirun -np $np -f /home/parallel/122033910188/spec-botsspar/iplist /home/parallel/122033910188/spec-botsspar/src/botsspar  -n $n -m $m -t $t -c"
mpirun -np $np -f /home/parallel/122033910188/spec-botsspar/iplist /home/parallel/122033910188/spec-botsspar/src/botsspar  -n $n -m $m -t $t -c
