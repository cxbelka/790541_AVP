run: build
	@./mulmatrix.out

build: asm link
	
	chmod +x mulmatrix.out

asm:
	gcc -fverbose-asm -mavx -mfma -O3 -S -g mulmatrix.c -o mulmatrix.S

link:
	gcc -c mulmatrix.S -o mulmatrix.obj
	gcc mulmatrix.obj -o mulmatrix.out
	objdump -d -S mulmatrix.out > mulmatrix.obj
#gcc -g -mavx -mfma -O3 mulmatrix.c -o mulmatrix.out