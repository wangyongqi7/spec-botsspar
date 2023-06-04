CC=mpicxx
include=-I common -I ./omp-tasks/sparselu/sparselu_single

all:botsspar
botsspar:bots_main.o bots_common.o sparselu.o
	${CC} -o botsspar -fopenmp bots_main.o bots_common.o sparselu.o
bots_main.o:
	${CC} -o bots_main.o -fopenmp -c ${include} common/bots_main.c
bots_common.o:
	${CC} -o bots_common.o -fopenmp -c ${include} common/bots_common.c
sparselu.o:
	${CC} -o sparselu.o -fopenmp -c ${include} ./omp-tasks/sparselu/sparselu_single/sparselu.c
clean:
	rm -f botsspar *.o