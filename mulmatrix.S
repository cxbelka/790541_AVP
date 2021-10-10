	.file	"mulmatrix.c"
# GNU C11 (Ubuntu 7.4.0-1ubuntu1~18.04.1) version 7.4.0 (x86_64-linux-gnu)
#	compiled by GNU C version 7.4.0, GMP version 6.1.2, MPFR version 4.0.1, MPC version 1.1.0, isl version isl-0.19-GMP

# GGC heuristics: --param ggc-min-expand=100 --param ggc-min-heapsize=131072
# options passed:  -imultiarch x86_64-linux-gnu mulmatrix.c -mavx -mfma
# -mtune=generic -march=x86-64 -auxbase-strip mulmatrix.S -g -O3
# -fverbose-asm -fstack-protector-strong -Wformat -Wformat-security
# options enabled:  -fPIC -fPIE -faggressive-loop-optimizations
# -falign-labels -fasynchronous-unwind-tables -fauto-inc-dec
# -fbranch-count-reg -fcaller-saves -fchkp-check-incomplete-type
# -fchkp-check-read -fchkp-check-write -fchkp-instrument-calls
# -fchkp-narrow-bounds -fchkp-optimize -fchkp-store-bounds
# -fchkp-use-static-bounds -fchkp-use-static-const-bounds
# -fchkp-use-wrappers -fcode-hoisting -fcombine-stack-adjustments -fcommon
# -fcompare-elim -fcprop-registers -fcrossjumping -fcse-follow-jumps
# -fdefer-pop -fdelete-null-pointer-checks -fdevirtualize
# -fdevirtualize-speculatively -fdwarf2-cfi-asm -fearly-inlining
# -feliminate-unused-debug-types -fexpensive-optimizations
# -fforward-propagate -ffp-int-builtin-inexact -ffunction-cse -fgcse
# -fgcse-after-reload -fgcse-lm -fgnu-runtime -fgnu-unique
# -fguess-branch-probability -fhoist-adjacent-loads -fident -fif-conversion
# -fif-conversion2 -findirect-inlining -finline -finline-atomics
# -finline-functions -finline-functions-called-once
# -finline-small-functions -fipa-bit-cp -fipa-cp -fipa-cp-clone -fipa-icf
# -fipa-icf-functions -fipa-icf-variables -fipa-profile -fipa-pure-const
# -fipa-ra -fipa-reference -fipa-sra -fipa-vrp -fira-hoist-pressure
# -fira-share-save-slots -fira-share-spill-slots
# -fisolate-erroneous-paths-dereference -fivopts -fkeep-static-consts
# -fleading-underscore -flifetime-dse -flra-remat -flto-odr-type-merging
# -fmath-errno -fmerge-constants -fmerge-debug-strings
# -fmove-loop-invariants -fomit-frame-pointer -foptimize-sibling-calls
# -foptimize-strlen -fpartial-inlining -fpeel-loops -fpeephole -fpeephole2
# -fplt -fpredictive-commoning -fprefetch-loop-arrays -free
# -freg-struct-return -freorder-blocks -freorder-functions
# -frerun-cse-after-loop -fsched-critical-path-heuristic
# -fsched-dep-count-heuristic -fsched-group-heuristic -fsched-interblock
# -fsched-last-insn-heuristic -fsched-rank-heuristic -fsched-spec
# -fsched-spec-insn-heuristic -fsched-stalled-insns-dep -fschedule-fusion
# -fschedule-insns2 -fsemantic-interposition -fshow-column -fshrink-wrap
# -fshrink-wrap-separate -fsigned-zeros -fsplit-ivs-in-unroller
# -fsplit-loops -fsplit-paths -fsplit-wide-types -fssa-backprop
# -fssa-phiopt -fstack-protector-strong -fstdarg-opt -fstore-merging
# -fstrict-aliasing -fstrict-overflow -fstrict-volatile-bitfields
# -fsync-libcalls -fthread-jumps -ftoplevel-reorder -ftrapping-math
# -ftree-bit-ccp -ftree-builtin-call-dce -ftree-ccp -ftree-ch
# -ftree-coalesce-vars -ftree-copy-prop -ftree-cselim -ftree-dce
# -ftree-dominator-opts -ftree-dse -ftree-forwprop -ftree-fre
# -ftree-loop-distribute-patterns -ftree-loop-if-convert -ftree-loop-im
# -ftree-loop-ivcanon -ftree-loop-optimize -ftree-loop-vectorize
# -ftree-parallelize-loops= -ftree-partial-pre -ftree-phiprop -ftree-pre
# -ftree-pta -ftree-reassoc -ftree-scev-cprop -ftree-sink
# -ftree-slp-vectorize -ftree-slsr -ftree-sra -ftree-switch-conversion
# -ftree-tail-merge -ftree-ter -ftree-vrp -funit-at-a-time -funswitch-loops
# -funwind-tables -fvar-tracking -fvar-tracking-assignments -fverbose-asm
# -fzero-initialized-in-bss -m128bit-long-double -m64 -m80387
# -malign-stringops -mavx -mavx256-split-unaligned-load
# -mavx256-split-unaligned-store -mfancy-math-387 -mfma -mfp-ret-in-387
# -mfxsr -mglibc -mieee-fp -mlong-double-80 -mmmx -mpopcnt -mpush-args
# -mred-zone -msse -msse2 -msse3 -msse4 -msse4.1 -msse4.2 -mssse3 -mstv
# -mtls-direct-seg-refs -mvzeroupper -mxsave

	.text
.Ltext0:
	.p2align 4,,15
	.globl	mulmat
	.type	mulmat, @function
mulmat:
.LFB4792:
	.file 1 "mulmatrix.c"
	.loc 1 20 0
	.cfi_startproc
.LVL0:
.LBB35:
.LBB36:
.LBB37:
# mulmatrix.c:28:                 C[i*CC+j] += A[i*AC+k]*B[k*BR+j];
	.loc 1 28 0
	movq	B(%rip), %rax	# B, B.3_14
	movq	A(%rip), %rdi	# A, ivtmp.51
	movq	C(%rip), %r9	# C, ivtmp.52
	leaq	8388608(%rdi), %r11	#, _87
	leaq	8388608(%rax), %r10	#, ivtmp.43
	leaq	8396800(%rax), %r8	#, _71
.LVL1:
.L2:
	movq	%r10, %rsi	# ivtmp.43, ivtmp.43
.LBE37:
.LBE36:
.LBE35:
# mulmatrix.c:20: void mulmat(void){
	.loc 1 20 0
	movq	%r9, %rcx	# ivtmp.52, ivtmp.41
.LVL2:
	.p2align 4,,10
	.p2align 3
.L4:
.LBB40:
.LBB39:
# mulmatrix.c:25:             C[i*CC+j] = 0;
	.loc 1 25 0
	vxorpd	%xmm0, %xmm0, %xmm0	# _22
	leaq	-8388608(%rsi), %rax	#, ivtmp.33
	movq	$0x000000000, (%rcx)	#, MEM[base: _74, offset: 0B]
.LVL3:
	movq	%rdi, %rdx	# ivtmp.51, ivtmp.32
.LVL4:
	.p2align 4,,10
	.p2align 3
.L3:
.LBB38:
# mulmatrix.c:28:                 C[i*CC+j] += A[i*AC+k]*B[k*BR+j];
	.loc 1 28 0 discriminator 3
	vmovsd	(%rdx), %xmm2	# MEM[base: _41, offset: 0B], tmp142
	addq	$8192, %rax	#, ivtmp.33
	addq	$8, %rdx	#, ivtmp.32
	vfmadd231sd	-8192(%rax), %xmm2, %xmm0	# MEM[base: _40, offset: 0B], tmp142, _22
# mulmatrix.c:26:             for (int k = 0; k < AC; k++)
	.loc 1 26 0 discriminator 3
	cmpq	%rsi, %rax	# ivtmp.43, ivtmp.33
# mulmatrix.c:28:                 C[i*CC+j] += A[i*AC+k]*B[k*BR+j];
	.loc 1 28 0 discriminator 3
	vmovsd	%xmm0, (%rcx)	# _22, MEM[base: _74, offset: 0B]
# mulmatrix.c:26:             for (int k = 0; k < AC; k++)
	.loc 1 26 0 discriminator 3
	jne	.L3	#,
	leaq	8(%rax), %rsi	#, ivtmp.43
	addq	$8, %rcx	#, ivtmp.41
.LBE38:
# mulmatrix.c:23:         for (int j = 0; j < BC; j++)
	.loc 1 23 0 discriminator 2
	cmpq	%rsi, %r8	# ivtmp.43, _71
	jne	.L4	#,
	addq	$8192, %rdi	#, ivtmp.51
	addq	$8192, %r9	#, ivtmp.52
.LBE39:
# mulmatrix.c:21:     for (int i = 0; i < AR; i++)
	.loc 1 21 0 discriminator 2
	cmpq	%r11, %rdi	# _87, ivtmp.51
	jne	.L2	#,
.LBE40:
# mulmatrix.c:32: }
	.loc 1 32 0
	rep ret
	.cfi_endproc
.LFE4792:
	.size	mulmat, .-mulmat
	.p2align 4,,15
	.globl	mulstr
	.type	mulstr, @function
mulstr:
.LFB4793:
	.loc 1 34 0
	.cfi_startproc
.LVL5:
	leaq	8(%rsp), %r10	#,
	.cfi_def_cfa 10, 0
	andq	$-32, %rsp	#,
	pushq	-8(%r10)	#
	pushq	%rbp	#
	.cfi_escape 0x10,0x6,0x2,0x76,0
	movq	%rsp, %rbp	#,
	pushq	%r15	#
	pushq	%r14	#
	pushq	%r13	#
	pushq	%r12	#
	pushq	%r10	#
	.cfi_escape 0xf,0x3,0x76,0x58,0x6
	.cfi_escape 0x10,0xf,0x2,0x76,0x78
	.cfi_escape 0x10,0xe,0x2,0x76,0x70
	.cfi_escape 0x10,0xd,0x2,0x76,0x68
	.cfi_escape 0x10,0xc,0x2,0x76,0x60
	pushq	%rbx	#
	.cfi_escape 0x10,0x3,0x2,0x76,0x50
.LBB41:
.LBB42:
# mulmatrix.c:36:         double* c = C + i*CC; //указатель на 1 элемент i-ой строки С
	.loc 1 36 0
	movq	C(%rip), %rax	# C, C.5_1
.LBB43:
.LBB44:
# mulmatrix.c:43:             double a = A[i*AC + k];//значение k-го элемента в i-ой строке  А
	.loc 1 43 0
	movq	A(%rip), %rbx	# A, A.7_12
# mulmatrix.c:42:             const double* b = B + k*BC;//указатель на 1 элемент строки В
	.loc 1 42 0
	movq	B(%rip), %r15	# B, B.6_8
# mulmatrix.c:43:             double a = A[i*AC + k];//значение k-го элемента в i-ой строке  А
	.loc 1 43 0
	movq	$0, -96(%rbp)	#, %sfp
	movq	$0, -88(%rbp)	#, %sfp
.LBE44:
.LBE43:
# mulmatrix.c:36:         double* c = C + i*CC; //указатель на 1 элемент i-ой строки С
	.loc 1 36 0
	movq	%rax, -104(%rbp)	# C.5_1, %sfp
.LBB52:
.LBB49:
# mulmatrix.c:43:             double a = A[i*AC + k];//значение k-го элемента в i-ой строке  А
	.loc 1 43 0
	movq	%rbx, -112(%rbp)	# A.7_12, %sfp
	movq	%rax, %r13	# C.5_1, ivtmp.109
	leaq	32(%rax), %rbx	#, ivtmp.111
	addq	$8388608, %rax	#, _228
	movq	%rax, -120(%rbp)	# _228, %sfp
	leaq	8388608(%r15), %rax	#, _178
	movq	%rbx, -80(%rbp)	# ivtmp.111, %sfp
	movq	%rax, -56(%rbp)	# _178, %sfp
.LVL6:
.L19:
	leaq	8(%r13), %rdi	#, tmp218
	movq	%r13, %rcx	# ivtmp.109, c
	xorl	%eax, %eax	# tmp315
	movq	$0, 0(%r13)	#,* ivtmp.109
	movq	$0, 8184(%r13)	#,
	movl	$1024, %ebx	#, niters.59
	andq	$-8, %rdi	#, tmp218
	movq	%r15, %rsi	# B.6_8, ivtmp.100
.LBE49:
.LBE52:
# mulmatrix.c:36:         double* c = C + i*CC; //указатель на 1 элемент i-ой строки С
	.loc 1 36 0
	xorl	%r11d, %r11d	# ivtmp.102
	subq	%rdi, %rcx	# tmp218, c
	addl	$8192, %ecx	#, tmp213
	shrl	$3, %ecx	#,
	rep stosq
	movq	%r13, %rax	# ivtmp.109, tmp221
	shrq	$3, %rax	#, tmp221
	negq	%rax	# tmp222
	movq	-96(%rbp), %rcx	# %sfp, ivtmp.113
	movq	-104(%rbp), %r14	# %sfp, C.5_1
	movl	%eax, %edi	# tmp222, prolog_loop_niters.57
	movq	-112(%rbp), %r12	# %sfp, ivtmp.99
	andl	$3, %edi	#, prolog_loop_niters.57
	subl	%edi, %ebx	# prolog_loop_niters.57, niters.59
	movl	%edi, %eax	# prolog_loop_niters.57, prolog_loop_adjusted_niters.58
	movl	%edi, -60(%rbp)	# prolog_loop_niters.57, %sfp
	movq	-88(%rbp), %rdi	# %sfp, ivtmp.112
	movl	%ebx, %r9d	# niters.59, bnd.60
	movl	%ebx, -64(%rbp)	# niters.59, %sfp
	andl	$-4, %ebx	#, niters_vector_mult_vf.61
	addq	%rcx, %r12	# ivtmp.113, ivtmp.99
	shrl	$2, %r9d	#, bnd.60
	leaq	(%rax,%rdi), %rdx	#, tmp224
	leaq	0(,%rax,8), %rdi	#, _149
	movq	%rdi, -72(%rbp)	# _149, %sfp
	addq	%rcx, %rdi	# ivtmp.113, tmp226
	leaq	(%r14,%rdx,8), %r8	#, vectp.65
	addq	%r14, %rdi	# C.5_1, vectp.73
	movl	%ebx, %r14d	# niters_vector_mult_vf.61, niters_vector_mult_vf.61
	.p2align 4,,10
	.p2align 3
.L18:
.LVL7:
	leaq	32(%rsi), %rax	#, tmp227
.LBB53:
.LBB50:
# mulmatrix.c:43:             double a = A[i*AC + k];//значение k-го элемента в i-ой строке  А
	.loc 1 43 0
	vmovsd	(%r12), %xmm1	# MEM[base: _192, offset: 0B], a
.LVL8:
	cmpq	%rax, %r13	# tmp227, ivtmp.109
	jnb	.L24	#,
	cmpq	-80(%rbp), %rsi	# %sfp, ivtmp.100
	jb	.L20	#,
.L24:
	movl	-60(%rbp), %eax	# %sfp, prolog_loop_niters.57
	testl	%eax, %eax	# prolog_loop_niters.57
	je	.L21	#,
.LBB45:
# mulmatrix.c:46:             c[j] += a * b[j];//формируем полусуммы
	.loc 1 46 0
	vmovsd	(%rsi), %xmm0	# MEM[base: _183, offset: 0B], tmp232
	cmpl	$1, %eax	#, prolog_loop_niters.57
	vfmadd213sd	0(%r13), %xmm1, %xmm0	# MEM[base: c_35, offset: 0B], a, tmp232
	vmovsd	%xmm0, 0(%r13)	# tmp232, MEM[base: c_35, offset: 0B]
.LVL9:
	je	.L22	#,
	vmovsd	8(%rsi), %xmm0	# MEM[base: _183, offset: 8B], tmp233
	cmpl	$3, %eax	#, prolog_loop_niters.57
	vfmadd213sd	8(%r13), %xmm1, %xmm0	# MEM[base: c_35, offset: 8B], a, tmp233
	vmovsd	%xmm0, 8(%r13)	# tmp233, MEM[base: c_35, offset: 8B]
.LVL10:
	jne	.L23	#,
	vmovsd	16(%rsi), %xmm0	# MEM[base: _183, offset: 16B], tmp234
	movl	$1021, %ebx	#, ivtmp_80
# mulmatrix.c:45:             for (int j = 0; j < CC; j++){//идём по строке С
	.loc 1 45 0
	movl	$3, %r10d	#, j
# mulmatrix.c:46:             c[j] += a * b[j];//формируем полусуммы
	.loc 1 46 0
	vfmadd213sd	16(%r13), %xmm1, %xmm0	# MEM[base: c_35, offset: 16B], a, tmp234
	vmovsd	%xmm0, 16(%r13)	# tmp234, MEM[base: c_35, offset: 16B]
.LVL11:
.L13:
	movq	-72(%rbp), %rax	# %sfp, _149
	vmovddup	%xmm1, %xmm2	# a, vect_cst__154
# mulmatrix.c:45:             for (int j = 0; j < CC; j++){//идём по строке С
	.loc 1 45 0
	xorl	%edx, %edx	# ivtmp.83
	leaq	(%rax,%r11), %rcx	#, tmp235
	vinsertf128	$1, %xmm2, %ymm2, %ymm2	# vect_cst__154, vect_cst__154
	xorl	%eax, %eax	# ivtmp.86
	addq	%r15, %rcx	# B.6_8, vectp.68
	.p2align 4,,10
	.p2align 3
.L15:
# mulmatrix.c:46:             c[j] += a * b[j];//формируем полусуммы
	.loc 1 46 0 discriminator 3
	vmovupd	(%rcx,%rax), %xmm0	# MEM[base: vectp.68_148, index: ivtmp.86_111, offset: 0B], MEM[base: vectp.68_148, index: ivtmp.86_111, offset: 0B]
	addl	$1, %edx	#, ivtmp.83
	vinsertf128	$0x1, 16(%rcx,%rax), %ymm0, %ymm0	# MEM[base: vectp.68_148, index: ivtmp.86_111, offset: 0B], MEM[base: vectp.68_148, index: ivtmp.86_111, offset: 0B], vect__22.69
	vfmadd213pd	(%r8,%rax), %ymm2, %ymm0	# MEM[base: vectp.65_140, index: ivtmp.86_111, offset: 0B], vect_cst__154, vect__24.71
	vmovapd	%ymm0, (%rdi,%rax)	# vect__24.71, MEM[base: vectp.73_157, index: ivtmp.86_111, offset: 0B]
	addq	$32, %rax	#, ivtmp.86
	cmpl	%r9d, %edx	# bnd.60, ivtmp.83
	jb	.L15	#,
	addl	%r14d, %r10d	# niters_vector_mult_vf.61, tmp.62
	subl	%r14d, %ebx	# niters_vector_mult_vf.61, tmp.63
	cmpl	%r14d, -64(%rbp)	# niters_vector_mult_vf.61, %sfp
	je	.L17	#,
.LVL12:
# mulmatrix.c:46:             c[j] += a * b[j];//формируем полусуммы
	.loc 1 46 0 is_stmt 0
	movslq	%r10d, %rax	# tmp.62, tmp.62
	salq	$3, %rax	#, _17
	leaq	0(%r13,%rax), %rdx	#, _18
	addq	%r15, %rax	# B.6_8, tmp241
# mulmatrix.c:45:             for (int j = 0; j < CC; j++){//идём по строке С
	.loc 1 45 0 is_stmt 1
	cmpl	$1, %ebx	#, tmp.63
# mulmatrix.c:46:             c[j] += a * b[j];//формируем полусуммы
	.loc 1 46 0
	vmovsd	(%rax,%r11), %xmm0	# *_33, tmp243
# mulmatrix.c:45:             for (int j = 0; j < CC; j++){//идём по строке С
	.loc 1 45 0
	leal	1(%r10), %eax	#, j
.LVL13:
# mulmatrix.c:46:             c[j] += a * b[j];//формируем полусуммы
	.loc 1 46 0
	vfmadd213sd	(%rdx), %xmm1, %xmm0	# *_18, a, tmp243
	vmovsd	%xmm0, (%rdx)	# tmp243, *_18
# mulmatrix.c:45:             for (int j = 0; j < CC; j++){//идём по строке С
	.loc 1 45 0
	je	.L17	#,
# mulmatrix.c:46:             c[j] += a * b[j];//формируем полусуммы
	.loc 1 46 0
	cltq
# mulmatrix.c:45:             for (int j = 0; j < CC; j++){//идём по строке С
	.loc 1 45 0
	addl	$2, %r10d	#, j
# mulmatrix.c:46:             c[j] += a * b[j];//формируем полусуммы
	.loc 1 46 0
	salq	$3, %rax	#, _167
.LVL14:
	leaq	0(%r13,%rax), %rdx	#, _168
	addq	%r15, %rax	# B.6_8, tmp246
# mulmatrix.c:45:             for (int j = 0; j < CC; j++){//идём по строке С
	.loc 1 45 0
	cmpl	$2, %ebx	#, tmp.63
# mulmatrix.c:46:             c[j] += a * b[j];//формируем полусуммы
	.loc 1 46 0
	vmovsd	(%rax,%r11), %xmm0	# *_171, tmp248
	vfmadd213sd	(%rdx), %xmm1, %xmm0	# *_168, a, tmp248
	vmovsd	%xmm0, (%rdx)	# tmp248, *_168
.LVL15:
# mulmatrix.c:45:             for (int j = 0; j < CC; j++){//идём по строке С
	.loc 1 45 0
	je	.L17	#,
# mulmatrix.c:46:             c[j] += a * b[j];//формируем полусуммы
	.loc 1 46 0
	movslq	%r10d, %r10	# j, j
	leaq	0(,%r10,8), %rax	#, _122
	leaq	0(%r13,%rax), %rdx	#, _123
	addq	%r15, %rax	# B.6_8, tmp250
	vmovsd	(%rdx), %xmm3	# *_123, tmp342
	vfmadd132sd	(%rax,%r11), %xmm3, %xmm1	# *_126, tmp342, tmp252
.LVL16:
	vmovsd	%xmm1, (%rdx)	# tmp252, *_123
.L17:
.LVL17:
	addq	$8, %r12	#, ivtmp.99
	addq	$8192, %rsi	#, ivtmp.100
	addq	$8192, %r11	#, ivtmp.102
.LVL18:
.LBE45:
.LBE50:
# mulmatrix.c:41:         for (int k = 0; k < AC; k++){//AC=BR, идём по столбцам B и элементам в строке А, k = номер столбца А/номер строки В
	.loc 1 41 0 discriminator 2
	cmpq	%rsi, -56(%rbp)	# ivtmp.100, %sfp
	jne	.L18	#,
	addq	$8192, %r13	#, ivtmp.109
.LVL19:
	addq	$8192, -80(%rbp)	#, %sfp
	addq	$1024, -88(%rbp)	#, %sfp
	addq	$8192, -96(%rbp)	#, %sfp
.LBE53:
.LBE42:
# mulmatrix.c:35:     for (int i = 0; i < AR; i++){//идём по строкам А и С, i = номер строки в А
	.loc 1 35 0 discriminator 2
	cmpq	-120(%rbp), %r13	# %sfp, ivtmp.109
	jne	.L19	#,
	vzeroupper
.LBE41:
# mulmatrix.c:51: }
	.loc 1 51 0
	popq	%rbx	#
	popq	%r10	#
	.cfi_remember_state
	.cfi_def_cfa 10, 0
	popq	%r12	#
	popq	%r13	#
.LVL20:
	popq	%r14	#
	popq	%r15	#
	popq	%rbp	#
	leaq	-8(%r10), %rsp	#,
	.cfi_def_cfa 7, 8
	ret
.LVL21:
	.p2align 4,,10
	.p2align 3
.L22:
	.cfi_restore_state
.LBB56:
.LBB55:
.LBB54:
.LBB51:
.LBB46:
# mulmatrix.c:46:             c[j] += a * b[j];//формируем полусуммы
	.loc 1 46 0
	movl	$1023, %ebx	#, ivtmp_80
# mulmatrix.c:45:             for (int j = 0; j < CC; j++){//идём по строке С
	.loc 1 45 0
	movl	$1, %r10d	#, j
	jmp	.L13	#
.LVL22:
.L21:
.LBE46:
# mulmatrix.c:43:             double a = A[i*AC + k];//значение k-го элемента в i-ой строке  А
	.loc 1 43 0
	movl	$1024, %ebx	#, ivtmp_80
.LBB47:
# mulmatrix.c:45:             for (int j = 0; j < CC; j++){//идём по строке С
	.loc 1 45 0
	xorl	%r10d, %r10d	# j
	jmp	.L13	#
.L20:
.LBE47:
# mulmatrix.c:43:             double a = A[i*AC + k];//значение k-го элемента в i-ой строке  А
	.loc 1 43 0
	xorl	%eax, %eax	# ivtmp.82
.LVL23:
	.p2align 4,,10
	.p2align 3
.L11:
.LBB48:
# mulmatrix.c:46:             c[j] += a * b[j];//формируем полусуммы
	.loc 1 46 0
	vmovsd	(%rsi,%rax), %xmm0	# MEM[base: _183, index: ivtmp.82_112, offset: 0B], tmp253
	vfmadd213sd	0(%r13,%rax), %xmm1, %xmm0	# MEM[base: c_35, index: ivtmp.82_112, offset: 0B], a, tmp253
	vmovsd	%xmm0, 0(%r13,%rax)	# tmp253, MEM[base: c_35, index: ivtmp.82_112, offset: 0B]
	addq	$8, %rax	#, ivtmp.82
# mulmatrix.c:45:             for (int j = 0; j < CC; j++){//идём по строке С
	.loc 1 45 0
	cmpq	$8192, %rax	#, ivtmp.82
	jne	.L11	#,
	jmp	.L17	#
.LVL24:
.L23:
# mulmatrix.c:46:             c[j] += a * b[j];//формируем полусуммы
	.loc 1 46 0
	movl	$1022, %ebx	#, ivtmp_80
# mulmatrix.c:45:             for (int j = 0; j < CC; j++){//идём по строке С
	.loc 1 45 0
	movl	$2, %r10d	#, j
	jmp	.L13	#
.LBE48:
.LBE51:
.LBE54:
.LBE55:
.LBE56:
	.cfi_endproc
.LFE4793:
	.size	mulstr, .-mulstr
	.p2align 4,,15
	.globl	mulvec
	.type	mulvec, @function
mulvec:
.LFB4794:
	.loc 1 53 0
	.cfi_startproc
.LVL25:
	leaq	8(%rsp), %r10	#,
	.cfi_def_cfa 10, 0
	andq	$-32, %rsp	#,
.LBB57:
.LBB58:
# /usr/include/x86_64-linux-gnu/bits/string_fortified.h:71:   return __builtin___memset_chk (__dest, __ch, __len, __bos0 (__dest));
	.file 2 "/usr/include/x86_64-linux-gnu/bits/string_fortified.h"
	.loc 2 71 0
	xorl	%esi, %esi	#
	movl	$8388608, %edx	#,
.LBE58:
.LBE57:
# mulmatrix.c:53: void mulvec(void){
	.loc 1 53 0
	pushq	-8(%r10)	#
	pushq	%rbp	#
	.cfi_escape 0x10,0x6,0x2,0x76,0
	movq	%rsp, %rbp	#,
	pushq	%r10	#
	.cfi_escape 0xf,0x3,0x76,0x78,0x6
	subq	$8, %rsp	#,
.LBB60:
.LBB59:
# /usr/include/x86_64-linux-gnu/bits/string_fortified.h:71:   return __builtin___memset_chk (__dest, __ch, __len, __bos0 (__dest));
	.loc 2 71 0
	movq	C(%rip), %rdi	# C,
	call	memset@PLT	#
.LVL26:
	xorl	%esi, %esi	# ivtmp.137
.LVL27:
.L39:
.LBE59:
.LBE60:
.LBB61:
.LBB62:
# mulmatrix.c:56:         double* c = C + i*CC; //указатель на 1 элемент i-ой строки С
	.loc 1 56 0
	movq	C(%rip), %rax	# C, c
	leaq	8192(%rsi), %r8	#, _26
	xorl	%edi, %edi	# ivtmp.136
	addq	%rsi, %rax	# ivtmp.137, c
.LVL28:
	leaq	32(%rax), %r9	#, ivtmp.126
	leaq	8224(%rax), %rcx	#, _64
.LVL29:
	.p2align 4,,10
	.p2align 3
.L38:
.LBB63:
.LBB64:
# mulmatrix.c:64:             __m256d a =  _mm256_broadcast_sd (A + i*AC + k);//значение k-го элемента в i-ой строке А множим в вектор
	.loc 1 64 0
	movq	A(%rip), %rax	# A, tmp125
	leaq	32(%rdi), %rdx	#, tmp127
	addq	B(%rip), %rdx	# B, ivtmp.127
	addq	%rsi, %rax	# ivtmp.137, tmp125
.LVL30:
.LBB65:
.LBB66:
# /usr/lib/gcc/x86_64-linux-gnu/7/include/avxintrin.h:728:   return (__m256d) __builtin_ia32_vbroadcastsd256 (__X);
	.file 3 "/usr/lib/gcc/x86_64-linux-gnu/7/include/avxintrin.h"
	.loc 3 728 0
	vbroadcastsd	(%rax), %ymm2	#, _37
.LVL31:
	movq	%r9, %rax	# ivtmp.126, ivtmp.126
.LVL32:
	.p2align 4,,10
	.p2align 3
.L37:
.LBE66:
.LBE65:
.LBB67:
.LBB68:
.LBB69:
# /usr/lib/gcc/x86_64-linux-gnu/7/include/avxintrin.h:885:   return *(__m256d_u *)__P;
	.loc 3 885 0 discriminator 3
	vmovupd	-32(%rdx), %xmm0	# MEM[base: _66, offset: -32B], MEM[base: _66, offset: -32B]
	addq	$64, %rax	#, ivtmp.126
	addq	$64, %rdx	#, ivtmp.127
.LBE69:
.LBE68:
.LBB71:
.LBB72:
	vmovupd	-96(%rax), %xmm1	# MEM[base: _68, offset: -32B], MEM[base: _68, offset: -32B]
.LBE72:
.LBE71:
.LBB74:
.LBB70:
	vinsertf128	$0x1, -80(%rdx), %ymm0, %ymm0	# MEM[base: _66, offset: -32B], MEM[base: _66, offset: -32B], tmp129
.LBE70:
.LBE74:
.LBB75:
.LBB73:
	vinsertf128	$0x1, -80(%rax), %ymm1, %ymm1	# MEM[base: _68, offset: -32B], MEM[base: _68, offset: -32B], tmp131
.LBE73:
.LBE75:
.LBB76:
.LBB77:
# /usr/lib/gcc/x86_64-linux-gnu/7/include/fmaintrin.h:49:   return (__m256d)__builtin_ia32_vfmaddpd256 ((__v4df)__A, (__v4df)__B,
	.file 4 "/usr/lib/gcc/x86_64-linux-gnu/7/include/fmaintrin.h"
	.loc 4 49 0 discriminator 3
	vfmadd132pd	%ymm2, %ymm1, %ymm0	# _37, tmp131, tmp128
.LVL33:
.LBE77:
.LBE76:
.LBB78:
.LBB79:
# /usr/lib/gcc/x86_64-linux-gnu/7/include/avxintrin.h:891:   *(__m256d_u *)__P = __A;
	.loc 3 891 0 discriminator 3
	vmovups	%xmm0, -96(%rax)	# tmp128, MEM[base: _68, offset: -32B]
	vextractf128	$0x1, %ymm0, -80(%rax)	# tmp128, MEM[base: _68, offset: -32B]
.LVL34:
.LBE79:
.LBE78:
.LBB80:
.LBB81:
# /usr/lib/gcc/x86_64-linux-gnu/7/include/avxintrin.h:885:   return *(__m256d_u *)__P;
	.loc 3 885 0 discriminator 3
	vmovupd	-64(%rax), %xmm1	# MEM[base: _68, offset: 0B], MEM[base: _68, offset: 0B]
.LBE81:
.LBE80:
.LBB83:
.LBB84:
	vmovupd	-64(%rdx), %xmm0	# MEM[base: _66, offset: 0B], MEM[base: _66, offset: 0B]
.LBE84:
.LBE83:
.LBB86:
.LBB82:
	vinsertf128	$0x1, -48(%rax), %ymm1, %ymm1	# MEM[base: _68, offset: 0B], MEM[base: _68, offset: 0B], tmp136
.LBE82:
.LBE86:
.LBB87:
.LBB85:
	vinsertf128	$0x1, -48(%rdx), %ymm0, %ymm0	# MEM[base: _66, offset: 0B], MEM[base: _66, offset: 0B], tmp134
.LBE85:
.LBE87:
.LBB88:
.LBB89:
# /usr/lib/gcc/x86_64-linux-gnu/7/include/fmaintrin.h:49:   return (__m256d)__builtin_ia32_vfmaddpd256 ((__v4df)__A, (__v4df)__B,
	.loc 4 49 0 discriminator 3
	vfmadd132pd	%ymm2, %ymm1, %ymm0	# _37, tmp136, tmp133
.LVL35:
.LBE89:
.LBE88:
.LBB90:
.LBB91:
# /usr/lib/gcc/x86_64-linux-gnu/7/include/avxintrin.h:891:   *(__m256d_u *)__P = __A;
	.loc 3 891 0 discriminator 3
	vmovups	%xmm0, -64(%rax)	# tmp133, MEM[base: _68, offset: 0B]
	vextractf128	$0x1, %ymm0, -48(%rax)	# tmp133, MEM[base: _68, offset: 0B]
.LBE91:
.LBE90:
# mulmatrix.c:66:             for (int j = 0; j < CC; j+=8){//идём по строке С
	.loc 1 66 0 discriminator 3
	cmpq	%rax, %rcx	# ivtmp.126, _64
	jne	.L37	#,
	addq	$8, %rsi	#, ivtmp.137
	addq	$8192, %rdi	#, ivtmp.136
.LBE67:
.LBE64:
# mulmatrix.c:61:         for (int k = 0; k < AC; k++){//AC=BR, идём по столбцам B и элементам в строке А, k = номер столбца А/номер строки В
	.loc 1 61 0 discriminator 2
	cmpq	%rsi, %r8	# ivtmp.137, _26
	jne	.L38	#,
.LBE63:
.LBE62:
# mulmatrix.c:55:     for (int i = 0; i < AR; i++){//идём по строкам А и С, i = номер строки в А
	.loc 1 55 0 discriminator 2
	cmpq	$8388608, %r8	#, ivtmp.137
	movq	%r8, %rsi	# _26, ivtmp.137
	jne	.L39	#,
	vzeroupper
.LBE61:
# mulmatrix.c:77: }
	.loc 1 77 0
	addq	$8, %rsp	#,
	popq	%r10	#
	.cfi_def_cfa 10, 0
	popq	%rbp	#
	leaq	-8(%r10), %rsp	#,
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4794:
	.size	mulvec, .-mulvec
	.section	.rodata.str1.8,"aMS",@progbits,1
	.align 8
.LC4:
	.string	"\320\222\321\200\320\265\320\274\321\217 \321\200\320\260\321\201\321\201\321\207\321\221\321\202\320\260 \320\274\320\260\321\202\321\200\320\270\321\206\321\213 \320\276\320\261\321\213\321\207\320\275\321\213\320\274 \321\201\320\277\320\276\321\201\320\276\320\261\320\276\320\274: %2.5f mksec \n"
	.align 8
.LC5:
	.string	"\320\222\321\200\320\265\320\274\321\217 \321\200\320\260\321\201\321\201\321\207\321\221\321\202\320\260 \320\274\320\260\321\202\321\200\320\270\321\206\321\213 \320\241\320\242\320\240\320\236\320\247\320\235\320\253\320\234 \321\201\320\277\320\276\321\201\320\276\320\261\320\276\320\274: %2.5f mksec \n"
	.align 8
.LC6:
	.string	"\320\222\321\200\320\265\320\274\321\217 \321\200\320\260\321\201\321\201\321\207\321\221\321\202\320\260 \320\274\320\260\321\202\321\200\320\270\321\206\321\213 \320\222\320\225\320\232\320\242\320\236\320\240\320\235\320\253\320\234 \321\201\320\277\320\276\321\201\320\276\320\261\320\276\320\274: %2.5f mksec \n"
	.section	.text.startup,"ax",@progbits
	.p2align 4,,15
	.globl	main
	.type	main, @function
main:
.LFB4795:
	.loc 1 80 0
	.cfi_startproc
.LVL36:
	pushq	%rbp	#
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	pushq	%rbx	#
	.cfi_def_cfa_offset 24
	.cfi_offset 3, -24
# mulmatrix.c:81:     A = malloc(sizeof(double)*AR*AC);
	.loc 1 81 0
	movl	$8388608, %edi	#,
.LVL37:
# mulmatrix.c:83:     C = malloc(sizeof(double)*CR*CC);
	.loc 1 83 0
	xorl	%ebx, %ebx	# ivtmp.161
# mulmatrix.c:80: {
	.loc 1 80 0
	subq	$56, %rsp	#,
	.cfi_def_cfa_offset 80
# mulmatrix.c:80: {
	.loc 1 80 0
	movq	%fs:40, %rax	#, tmp196
	movq	%rax, 40(%rsp)	# tmp196, D.29275
	xorl	%eax, %eax	# tmp196
# mulmatrix.c:81:     A = malloc(sizeof(double)*AR*AC);
	.loc 1 81 0
	call	malloc@PLT	#
.LVL38:
# mulmatrix.c:82:     B = malloc(sizeof(double)*BR*BC);
	.loc 1 82 0
	movl	$8388608, %edi	#,
# mulmatrix.c:81:     A = malloc(sizeof(double)*AR*AC);
	.loc 1 81 0
	movq	%rax, A(%rip)	# tmp143, A
# mulmatrix.c:82:     B = malloc(sizeof(double)*BR*BC);
	.loc 1 82 0
	call	malloc@PLT	#
.LVL39:
# mulmatrix.c:83:     C = malloc(sizeof(double)*CR*CC);
	.loc 1 83 0
	movl	$8388608, %edi	#,
# mulmatrix.c:82:     B = malloc(sizeof(double)*BR*BC);
	.loc 1 82 0
	movq	%rax, B(%rip)	# tmp144, B
# mulmatrix.c:83:     C = malloc(sizeof(double)*CR*CC);
	.loc 1 83 0
	call	malloc@PLT	#
.LVL40:
	movq	%rax, C(%rip)	# tmp145, C
.LVL41:
	.p2align 4,,10
	.p2align 3
.L45:
.LBB92:
# mulmatrix.c:87:         A[i] = drand48()*5; 
	.loc 1 87 0 discriminator 3
	call	drand48@PLT	#
.LVL42:
	vmulsd	.LC1(%rip), %xmm0, %xmm0	#, _4, tmp147
	movq	A(%rip), %rax	# A, A
	vmovsd	%xmm0, (%rax,%rbx)	# tmp147, *_8
	addq	$8, %rbx	#, ivtmp.161
# mulmatrix.c:85:     for (int i = 0; i < AC*AR; i++)
	.loc 1 85 0 discriminator 3
	cmpq	$8388608, %rbx	#, ivtmp.161
	jne	.L45	#,
	xorl	%ebx, %ebx	# ivtmp.154
	.p2align 4,,10
	.p2align 3
.L46:
.LBE92:
.LBB93:
# mulmatrix.c:92:         B[i] = drand48()*5; 
	.loc 1 92 0 discriminator 3
	call	drand48@PLT	#
.LVL43:
	vmulsd	.LC1(%rip), %xmm0, %xmm0	#, _10, tmp150
	movq	B(%rip), %rax	# B, B
	vmovsd	%xmm0, (%rax,%rbx)	# tmp150, *_14
.LVL44:
	addq	$8, %rbx	#, ivtmp.154
# mulmatrix.c:90:     for (int i = 0; i <BC*BR; i++)
	.loc 1 90 0 discriminator 3
	cmpq	$8388608, %rbx	#, ivtmp.154
	jne	.L46	#,
.LBE93:
# mulmatrix.c:98:     clock_gettime(CLOCK_REALTIME, &t0);
	.loc 1 98 0
	movq	%rsp, %rbp	#, tmp152
# mulmatrix.c:102:     clock_gettime(CLOCK_REALTIME, &t1);
	.loc 1 102 0
	leaq	16(%rsp), %rbx	#, tmp153
# mulmatrix.c:98:     clock_gettime(CLOCK_REALTIME, &t0);
	.loc 1 98 0
	xorl	%edi, %edi	#
	movq	%rbp, %rsi	# tmp152,
	call	clock_gettime@PLT	#
.LVL45:
# mulmatrix.c:100:     mulmat();
	.loc 1 100 0
	call	mulmat	#
.LVL46:
# mulmatrix.c:102:     clock_gettime(CLOCK_REALTIME, &t1);
	.loc 1 102 0
	xorl	%edi, %edi	#
	movq	%rbx, %rsi	# tmp153,
	call	clock_gettime@PLT	#
.LVL47:
# mulmatrix.c:104:     tres1 = ((t1.tv_sec - t0.tv_sec)*10e9 + (t1.tv_nsec - t0.tv_nsec))/10e6;
	.loc 1 104 0
	movq	16(%rsp), %rax	# t1.tv_sec, t1.tv_sec
	subq	(%rsp), %rax	# t0.tv_sec, tmp154
.LVL48:
.LBB94:
.LBB95:
# /usr/include/x86_64-linux-gnu/bits/stdio2.h:104:   return __printf_chk (__USE_FORTIFY_LEVEL - 1, __fmt, __va_arg_pack ());
	.file 5 "/usr/include/x86_64-linux-gnu/bits/stdio2.h"
	.loc 5 104 0
	leaq	.LC4(%rip), %rsi	#,
.LBE95:
.LBE94:
# mulmatrix.c:104:     tres1 = ((t1.tv_sec - t0.tv_sec)*10e9 + (t1.tv_nsec - t0.tv_nsec))/10e6;
	.loc 1 104 0
	vxorpd	%xmm0, %xmm0, %xmm0	# tmp156
.LBB99:
.LBB96:
# /usr/include/x86_64-linux-gnu/bits/stdio2.h:104:   return __printf_chk (__USE_FORTIFY_LEVEL - 1, __fmt, __va_arg_pack ());
	.loc 5 104 0
	movl	$1, %edi	#,
.LBE96:
.LBE99:
# mulmatrix.c:104:     tres1 = ((t1.tv_sec - t0.tv_sec)*10e9 + (t1.tv_nsec - t0.tv_nsec))/10e6;
	.loc 1 104 0
	vxorpd	%xmm1, %xmm1, %xmm1	# tmp159
	vcvtsi2sdq	%rax, %xmm0, %xmm0	# tmp154, tmp156, tmp156
	movq	24(%rsp), %rax	# t1.tv_nsec, t1.tv_nsec
.LVL49:
	subq	8(%rsp), %rax	# t0.tv_nsec, tmp157
.LVL50:
	vcvtsi2sdq	%rax, %xmm1, %xmm1	# tmp157, tmp159, tmp159
	vfmadd132sd	.LC2(%rip), %xmm1, %xmm0	#, tmp159, tmp160
	vdivsd	.LC3(%rip), %xmm0, %xmm0	#, tmp160, tmp162
.LBB100:
.LBB97:
# /usr/include/x86_64-linux-gnu/bits/stdio2.h:104:   return __printf_chk (__USE_FORTIFY_LEVEL - 1, __fmt, __va_arg_pack ());
	.loc 5 104 0
	movl	$1, %eax	#,
.LVL51:
.LBE97:
.LBE100:
# mulmatrix.c:104:     tres1 = ((t1.tv_sec - t0.tv_sec)*10e9 + (t1.tv_nsec - t0.tv_nsec))/10e6;
	.loc 1 104 0
	vcvtsd2ss	%xmm0, %xmm0, %xmm0	# tmp162, tres1
# mulmatrix.c:105:     printf("Время рассчёта матрицы обычным способом: %2.5f mksec \n", tres1);
	.loc 1 105 0
	vcvtss2sd	%xmm0, %xmm0, %xmm0	# tres1, tmp165
.LBB101:
.LBB98:
# /usr/include/x86_64-linux-gnu/bits/stdio2.h:104:   return __printf_chk (__USE_FORTIFY_LEVEL - 1, __fmt, __va_arg_pack ());
	.loc 5 104 0
	call	__printf_chk@PLT	#
.LVL52:
.LBE98:
.LBE101:
# mulmatrix.c:107:     clock_gettime(CLOCK_REALTIME, &t0);
	.loc 1 107 0
	xorl	%edi, %edi	#
	movq	%rbp, %rsi	# tmp152,
	call	clock_gettime@PLT	#
.LVL53:
# mulmatrix.c:109:     mulstr();
	.loc 1 109 0
	call	mulstr	#
.LVL54:
# mulmatrix.c:111:     clock_gettime(CLOCK_REALTIME, &t1);
	.loc 1 111 0
	xorl	%edi, %edi	#
	movq	%rbx, %rsi	# tmp153,
	call	clock_gettime@PLT	#
.LVL55:
# mulmatrix.c:113:     tres2 = ((t1.tv_sec - t0.tv_sec)*10e9 + (t1.tv_nsec - t0.tv_nsec))/10e6;
	.loc 1 113 0
	movq	16(%rsp), %rax	# t1.tv_sec, t1.tv_sec
	subq	(%rsp), %rax	# t0.tv_sec, tmp168
.LVL56:
.LBB102:
.LBB103:
# /usr/include/x86_64-linux-gnu/bits/stdio2.h:104:   return __printf_chk (__USE_FORTIFY_LEVEL - 1, __fmt, __va_arg_pack ());
	.loc 5 104 0
	leaq	.LC5(%rip), %rsi	#,
.LBE103:
.LBE102:
# mulmatrix.c:113:     tres2 = ((t1.tv_sec - t0.tv_sec)*10e9 + (t1.tv_nsec - t0.tv_nsec))/10e6;
	.loc 1 113 0
	vxorpd	%xmm0, %xmm0, %xmm0	# tmp170
.LBB107:
.LBB104:
# /usr/include/x86_64-linux-gnu/bits/stdio2.h:104:   return __printf_chk (__USE_FORTIFY_LEVEL - 1, __fmt, __va_arg_pack ());
	.loc 5 104 0
	movl	$1, %edi	#,
.LBE104:
.LBE107:
# mulmatrix.c:113:     tres2 = ((t1.tv_sec - t0.tv_sec)*10e9 + (t1.tv_nsec - t0.tv_nsec))/10e6;
	.loc 1 113 0
	vxorpd	%xmm1, %xmm1, %xmm1	# tmp173
	vcvtsi2sdq	%rax, %xmm0, %xmm0	# tmp168, tmp170, tmp170
	movq	24(%rsp), %rax	# t1.tv_nsec, t1.tv_nsec
.LVL57:
	subq	8(%rsp), %rax	# t0.tv_nsec, tmp171
.LVL58:
	vcvtsi2sdq	%rax, %xmm1, %xmm1	# tmp171, tmp173, tmp173
	vfmadd132sd	.LC2(%rip), %xmm1, %xmm0	#, tmp173, tmp174
	vdivsd	.LC3(%rip), %xmm0, %xmm0	#, tmp174, tmp176
.LBB108:
.LBB105:
# /usr/include/x86_64-linux-gnu/bits/stdio2.h:104:   return __printf_chk (__USE_FORTIFY_LEVEL - 1, __fmt, __va_arg_pack ());
	.loc 5 104 0
	movl	$1, %eax	#,
.LVL59:
.LBE105:
.LBE108:
# mulmatrix.c:113:     tres2 = ((t1.tv_sec - t0.tv_sec)*10e9 + (t1.tv_nsec - t0.tv_nsec))/10e6;
	.loc 1 113 0
	vcvtsd2ss	%xmm0, %xmm0, %xmm0	# tmp176, tres2
# mulmatrix.c:114:     printf("Время рассчёта матрицы СТРОЧНЫМ способом: %2.5f mksec \n", tres2);
	.loc 1 114 0
	vcvtss2sd	%xmm0, %xmm0, %xmm0	# tres2, tmp179
.LBB109:
.LBB106:
# /usr/include/x86_64-linux-gnu/bits/stdio2.h:104:   return __printf_chk (__USE_FORTIFY_LEVEL - 1, __fmt, __va_arg_pack ());
	.loc 5 104 0
	call	__printf_chk@PLT	#
.LVL60:
.LBE106:
.LBE109:
# mulmatrix.c:116:     clock_gettime(CLOCK_REALTIME, &t0);
	.loc 1 116 0
	xorl	%edi, %edi	#
	movq	%rbp, %rsi	# tmp152,
	call	clock_gettime@PLT	#
.LVL61:
# mulmatrix.c:118:     mulvec();
	.loc 1 118 0
	call	mulvec	#
.LVL62:
# mulmatrix.c:120:     clock_gettime(CLOCK_REALTIME, &t1);
	.loc 1 120 0
	xorl	%edi, %edi	#
	movq	%rbx, %rsi	# tmp153,
	call	clock_gettime@PLT	#
.LVL63:
# mulmatrix.c:122:     tres3 = ((t1.tv_sec - t0.tv_sec)*10e9 + (t1.tv_nsec - t0.tv_nsec))/10e6;
	.loc 1 122 0
	movq	16(%rsp), %rax	# t1.tv_sec, t1.tv_sec
	subq	(%rsp), %rax	# t0.tv_sec, tmp182
.LVL64:
.LBB110:
.LBB111:
# /usr/include/x86_64-linux-gnu/bits/stdio2.h:104:   return __printf_chk (__USE_FORTIFY_LEVEL - 1, __fmt, __va_arg_pack ());
	.loc 5 104 0
	leaq	.LC6(%rip), %rsi	#,
.LBE111:
.LBE110:
# mulmatrix.c:122:     tres3 = ((t1.tv_sec - t0.tv_sec)*10e9 + (t1.tv_nsec - t0.tv_nsec))/10e6;
	.loc 1 122 0
	vxorpd	%xmm0, %xmm0, %xmm0	# tmp184
.LBB115:
.LBB112:
# /usr/include/x86_64-linux-gnu/bits/stdio2.h:104:   return __printf_chk (__USE_FORTIFY_LEVEL - 1, __fmt, __va_arg_pack ());
	.loc 5 104 0
	movl	$1, %edi	#,
.LBE112:
.LBE115:
# mulmatrix.c:122:     tres3 = ((t1.tv_sec - t0.tv_sec)*10e9 + (t1.tv_nsec - t0.tv_nsec))/10e6;
	.loc 1 122 0
	vxorpd	%xmm1, %xmm1, %xmm1	# tmp187
	vcvtsi2sdq	%rax, %xmm0, %xmm0	# tmp182, tmp184, tmp184
	movq	24(%rsp), %rax	# t1.tv_nsec, t1.tv_nsec
.LVL65:
	subq	8(%rsp), %rax	# t0.tv_nsec, tmp185
.LVL66:
	vcvtsi2sdq	%rax, %xmm1, %xmm1	# tmp185, tmp187, tmp187
	vfmadd132sd	.LC2(%rip), %xmm1, %xmm0	#, tmp187, tmp188
	vdivsd	.LC3(%rip), %xmm0, %xmm0	#, tmp188, tmp190
.LBB116:
.LBB113:
# /usr/include/x86_64-linux-gnu/bits/stdio2.h:104:   return __printf_chk (__USE_FORTIFY_LEVEL - 1, __fmt, __va_arg_pack ());
	.loc 5 104 0
	movl	$1, %eax	#,
.LVL67:
.LBE113:
.LBE116:
# mulmatrix.c:122:     tres3 = ((t1.tv_sec - t0.tv_sec)*10e9 + (t1.tv_nsec - t0.tv_nsec))/10e6;
	.loc 1 122 0
	vcvtsd2ss	%xmm0, %xmm0, %xmm0	# tmp190, tres3
# mulmatrix.c:123:     printf("Время рассчёта матрицы ВЕКТОРНЫМ способом: %2.5f mksec \n", tres3);
	.loc 1 123 0
	vcvtss2sd	%xmm0, %xmm0, %xmm0	# tres3, tmp193
.LBB117:
.LBB114:
# /usr/include/x86_64-linux-gnu/bits/stdio2.h:104:   return __printf_chk (__USE_FORTIFY_LEVEL - 1, __fmt, __va_arg_pack ());
	.loc 5 104 0
	call	__printf_chk@PLT	#
.LVL68:
.LBE114:
.LBE117:
# mulmatrix.c:128: }
	.loc 1 128 0
	xorl	%eax, %eax	#
	movq	40(%rsp), %rdx	# D.29275, tmp197
	xorq	%fs:40, %rdx	#, tmp197
	jne	.L52	#,
	addq	$56, %rsp	#,
	.cfi_remember_state
	.cfi_def_cfa_offset 24
	popq	%rbx	#
	.cfi_def_cfa_offset 16
	popq	%rbp	#
	.cfi_def_cfa_offset 8
	ret
.L52:
	.cfi_restore_state
	call	__stack_chk_fail@PLT	#
.LVL69:
	.cfi_endproc
.LFE4795:
	.size	main, .-main
	.comm	C,8,8
	.comm	B,8,8
	.comm	A,8,8
	.section	.rodata.cst8,"aM",@progbits,8
	.align 8
.LC1:
	.long	0
	.long	1075052544
	.align 8
.LC2:
	.long	536870912
	.long	1107468383
	.align 8
.LC3:
	.long	0
	.long	1097011920
	.text
.Letext0:
	.file 6 "/usr/lib/gcc/x86_64-linux-gnu/7/include/stddef.h"
	.file 7 "/usr/include/x86_64-linux-gnu/bits/types.h"
	.file 8 "/usr/include/x86_64-linux-gnu/bits/libio.h"
	.file 9 "/usr/include/stdio.h"
	.file 10 "/usr/include/x86_64-linux-gnu/bits/sys_errlist.h"
	.file 11 "/usr/include/x86_64-linux-gnu/bits/types/struct_timespec.h"
	.file 12 "/usr/include/time.h"
	.file 13 "/usr/include/malloc.h"
	.file 14 "/usr/include/stdlib.h"
	.file 15 "<built-in>"
	.section	.debug_info,"",@progbits
.Ldebug_info0:
	.long	0xc59
	.value	0x4
	.long	.Ldebug_abbrev0
	.byte	0x8
	.uleb128 0x1
	.long	.LASF99
	.byte	0xc
	.long	.LASF100
	.long	.LASF101
	.long	.Ldebug_ranges0+0x3c0
	.quad	0
	.long	.Ldebug_line0
	.uleb128 0x2
	.long	.LASF7
	.byte	0x6
	.byte	0xd8
	.long	0x34
	.uleb128 0x3
	.byte	0x8
	.byte	0x7
	.long	.LASF0
	.uleb128 0x3
	.byte	0x1
	.byte	0x8
	.long	.LASF1
	.uleb128 0x3
	.byte	0x2
	.byte	0x7
	.long	.LASF2
	.uleb128 0x3
	.byte	0x4
	.byte	0x7
	.long	.LASF3
	.uleb128 0x3
	.byte	0x1
	.byte	0x6
	.long	.LASF4
	.uleb128 0x3
	.byte	0x2
	.byte	0x5
	.long	.LASF5
	.uleb128 0x4
	.byte	0x4
	.byte	0x5
	.string	"int"
	.uleb128 0x3
	.byte	0x8
	.byte	0x5
	.long	.LASF6
	.uleb128 0x2
	.long	.LASF8
	.byte	0x7
	.byte	0x8c
	.long	0x65
	.uleb128 0x2
	.long	.LASF9
	.byte	0x7
	.byte	0x8d
	.long	0x65
	.uleb128 0x2
	.long	.LASF10
	.byte	0x7
	.byte	0x94
	.long	0x65
	.uleb128 0x5
	.byte	0x8
	.uleb128 0x2
	.long	.LASF11
	.byte	0x7
	.byte	0xb8
	.long	0x65
	.uleb128 0x6
	.byte	0x8
	.long	0xa0
	.uleb128 0x3
	.byte	0x1
	.byte	0x6
	.long	.LASF12
	.uleb128 0x7
	.long	0xa0
	.uleb128 0x8
	.long	.LASF42
	.byte	0xd8
	.byte	0x8
	.byte	0xf5
	.long	0x22c
	.uleb128 0x9
	.long	.LASF13
	.byte	0x8
	.byte	0xf6
	.long	0x5e
	.byte	0
	.uleb128 0x9
	.long	.LASF14
	.byte	0x8
	.byte	0xfb
	.long	0x9a
	.byte	0x8
	.uleb128 0x9
	.long	.LASF15
	.byte	0x8
	.byte	0xfc
	.long	0x9a
	.byte	0x10
	.uleb128 0x9
	.long	.LASF16
	.byte	0x8
	.byte	0xfd
	.long	0x9a
	.byte	0x18
	.uleb128 0x9
	.long	.LASF17
	.byte	0x8
	.byte	0xfe
	.long	0x9a
	.byte	0x20
	.uleb128 0x9
	.long	.LASF18
	.byte	0x8
	.byte	0xff
	.long	0x9a
	.byte	0x28
	.uleb128 0xa
	.long	.LASF19
	.byte	0x8
	.value	0x100
	.long	0x9a
	.byte	0x30
	.uleb128 0xa
	.long	.LASF20
	.byte	0x8
	.value	0x101
	.long	0x9a
	.byte	0x38
	.uleb128 0xa
	.long	.LASF21
	.byte	0x8
	.value	0x102
	.long	0x9a
	.byte	0x40
	.uleb128 0xa
	.long	.LASF22
	.byte	0x8
	.value	0x104
	.long	0x9a
	.byte	0x48
	.uleb128 0xa
	.long	.LASF23
	.byte	0x8
	.value	0x105
	.long	0x9a
	.byte	0x50
	.uleb128 0xa
	.long	.LASF24
	.byte	0x8
	.value	0x106
	.long	0x9a
	.byte	0x58
	.uleb128 0xa
	.long	.LASF25
	.byte	0x8
	.value	0x108
	.long	0x264
	.byte	0x60
	.uleb128 0xa
	.long	.LASF26
	.byte	0x8
	.value	0x10a
	.long	0x26a
	.byte	0x68
	.uleb128 0xa
	.long	.LASF27
	.byte	0x8
	.value	0x10c
	.long	0x5e
	.byte	0x70
	.uleb128 0xa
	.long	.LASF28
	.byte	0x8
	.value	0x110
	.long	0x5e
	.byte	0x74
	.uleb128 0xa
	.long	.LASF29
	.byte	0x8
	.value	0x112
	.long	0x6c
	.byte	0x78
	.uleb128 0xa
	.long	.LASF30
	.byte	0x8
	.value	0x116
	.long	0x42
	.byte	0x80
	.uleb128 0xa
	.long	.LASF31
	.byte	0x8
	.value	0x117
	.long	0x50
	.byte	0x82
	.uleb128 0xa
	.long	.LASF32
	.byte	0x8
	.value	0x118
	.long	0x270
	.byte	0x83
	.uleb128 0xa
	.long	.LASF33
	.byte	0x8
	.value	0x11c
	.long	0x280
	.byte	0x88
	.uleb128 0xa
	.long	.LASF34
	.byte	0x8
	.value	0x125
	.long	0x77
	.byte	0x90
	.uleb128 0xa
	.long	.LASF35
	.byte	0x8
	.value	0x12d
	.long	0x8d
	.byte	0x98
	.uleb128 0xa
	.long	.LASF36
	.byte	0x8
	.value	0x12e
	.long	0x8d
	.byte	0xa0
	.uleb128 0xa
	.long	.LASF37
	.byte	0x8
	.value	0x12f
	.long	0x8d
	.byte	0xa8
	.uleb128 0xa
	.long	.LASF38
	.byte	0x8
	.value	0x130
	.long	0x8d
	.byte	0xb0
	.uleb128 0xa
	.long	.LASF39
	.byte	0x8
	.value	0x132
	.long	0x29
	.byte	0xb8
	.uleb128 0xa
	.long	.LASF40
	.byte	0x8
	.value	0x133
	.long	0x5e
	.byte	0xc0
	.uleb128 0xa
	.long	.LASF41
	.byte	0x8
	.value	0x135
	.long	0x286
	.byte	0xc4
	.byte	0
	.uleb128 0xb
	.long	.LASF102
	.byte	0x8
	.byte	0x9a
	.uleb128 0x8
	.long	.LASF43
	.byte	0x18
	.byte	0x8
	.byte	0xa0
	.long	0x264
	.uleb128 0x9
	.long	.LASF44
	.byte	0x8
	.byte	0xa1
	.long	0x264
	.byte	0
	.uleb128 0x9
	.long	.LASF45
	.byte	0x8
	.byte	0xa2
	.long	0x26a
	.byte	0x8
	.uleb128 0x9
	.long	.LASF46
	.byte	0x8
	.byte	0xa6
	.long	0x5e
	.byte	0x10
	.byte	0
	.uleb128 0x6
	.byte	0x8
	.long	0x233
	.uleb128 0x6
	.byte	0x8
	.long	0xac
	.uleb128 0xc
	.long	0xa0
	.long	0x280
	.uleb128 0xd
	.long	0x34
	.byte	0
	.byte	0
	.uleb128 0x6
	.byte	0x8
	.long	0x22c
	.uleb128 0xc
	.long	0xa0
	.long	0x296
	.uleb128 0xd
	.long	0x34
	.byte	0x13
	.byte	0
	.uleb128 0xe
	.long	.LASF103
	.uleb128 0xf
	.long	.LASF47
	.byte	0x8
	.value	0x13f
	.long	0x296
	.uleb128 0xf
	.long	.LASF48
	.byte	0x8
	.value	0x140
	.long	0x296
	.uleb128 0xf
	.long	.LASF49
	.byte	0x8
	.value	0x141
	.long	0x296
	.uleb128 0x6
	.byte	0x8
	.long	0xa7
	.uleb128 0x7
	.long	0x2bf
	.uleb128 0x10
	.long	0x2bf
	.uleb128 0x11
	.long	.LASF50
	.byte	0x9
	.byte	0x87
	.long	0x26a
	.uleb128 0x11
	.long	.LASF51
	.byte	0x9
	.byte	0x88
	.long	0x26a
	.uleb128 0x11
	.long	.LASF52
	.byte	0x9
	.byte	0x89
	.long	0x26a
	.uleb128 0x11
	.long	.LASF53
	.byte	0xa
	.byte	0x1a
	.long	0x5e
	.uleb128 0xc
	.long	0x2c5
	.long	0x306
	.uleb128 0x12
	.byte	0
	.uleb128 0x7
	.long	0x2fb
	.uleb128 0x11
	.long	.LASF54
	.byte	0xa
	.byte	0x1b
	.long	0x306
	.uleb128 0x8
	.long	.LASF55
	.byte	0x10
	.byte	0xb
	.byte	0x8
	.long	0x33b
	.uleb128 0x9
	.long	.LASF56
	.byte	0xb
	.byte	0xa
	.long	0x82
	.byte	0
	.uleb128 0x9
	.long	.LASF57
	.byte	0xb
	.byte	0xb
	.long	0x8f
	.byte	0x8
	.byte	0
	.uleb128 0xc
	.long	0x9a
	.long	0x34b
	.uleb128 0xd
	.long	0x34
	.byte	0x1
	.byte	0
	.uleb128 0x11
	.long	.LASF58
	.byte	0xc
	.byte	0x9f
	.long	0x33b
	.uleb128 0x11
	.long	.LASF59
	.byte	0xc
	.byte	0xa0
	.long	0x5e
	.uleb128 0x11
	.long	.LASF60
	.byte	0xc
	.byte	0xa1
	.long	0x65
	.uleb128 0x11
	.long	.LASF61
	.byte	0xc
	.byte	0xa6
	.long	0x33b
	.uleb128 0x11
	.long	.LASF62
	.byte	0xc
	.byte	0xae
	.long	0x5e
	.uleb128 0x11
	.long	.LASF63
	.byte	0xc
	.byte	0xaf
	.long	0x65
	.uleb128 0x2
	.long	.LASF64
	.byte	0x6
	.byte	0x95
	.long	0x65
	.uleb128 0x3
	.byte	0x8
	.byte	0x5
	.long	.LASF65
	.uleb128 0x3
	.byte	0x10
	.byte	0x4
	.long	.LASF66
	.uleb128 0x13
	.long	0x8d
	.long	0x3b5
	.uleb128 0x14
	.long	0x38d
	.byte	0
	.uleb128 0x11
	.long	.LASF67
	.byte	0xd
	.byte	0x4c
	.long	0x3c0
	.uleb128 0x6
	.byte	0x8
	.long	0x3a6
	.uleb128 0x15
	.long	0x3d6
	.uleb128 0x14
	.long	0x8d
	.uleb128 0x14
	.long	0x3d6
	.byte	0
	.uleb128 0x6
	.byte	0x8
	.long	0x3dc
	.uleb128 0x16
	.uleb128 0x11
	.long	.LASF68
	.byte	0xd
	.byte	0x8f
	.long	0x3ee
	.uleb128 0x6
	.byte	0x8
	.long	0x3c6
	.uleb128 0x17
	.long	0x3e8
	.uleb128 0x13
	.long	0x8d
	.long	0x407
	.uleb128 0x14
	.long	0x29
	.uleb128 0x14
	.long	0x3d6
	.byte	0
	.uleb128 0x11
	.long	.LASF69
	.byte	0xd
	.byte	0x92
	.long	0x418
	.uleb128 0x6
	.byte	0x8
	.long	0x3f3
	.uleb128 0x17
	.long	0x412
	.uleb128 0x13
	.long	0x8d
	.long	0x436
	.uleb128 0x14
	.long	0x8d
	.uleb128 0x14
	.long	0x29
	.uleb128 0x14
	.long	0x3d6
	.byte	0
	.uleb128 0x11
	.long	.LASF70
	.byte	0xd
	.byte	0x95
	.long	0x447
	.uleb128 0x6
	.byte	0x8
	.long	0x41d
	.uleb128 0x17
	.long	0x441
	.uleb128 0x13
	.long	0x8d
	.long	0x465
	.uleb128 0x14
	.long	0x29
	.uleb128 0x14
	.long	0x29
	.uleb128 0x14
	.long	0x3d6
	.byte	0
	.uleb128 0x11
	.long	.LASF71
	.byte	0xd
	.byte	0x99
	.long	0x476
	.uleb128 0x6
	.byte	0x8
	.long	0x44c
	.uleb128 0x17
	.long	0x470
	.uleb128 0x18
	.uleb128 0x11
	.long	.LASF72
	.byte	0xd
	.byte	0x9d
	.long	0x48d
	.uleb128 0x6
	.byte	0x8
	.long	0x47b
	.uleb128 0x17
	.long	0x487
	.uleb128 0x3
	.byte	0x8
	.byte	0x7
	.long	.LASF73
	.uleb128 0x3
	.byte	0x4
	.byte	0x4
	.long	.LASF74
	.uleb128 0x3
	.byte	0x8
	.byte	0x4
	.long	.LASF75
	.uleb128 0x7
	.long	0x4a0
	.uleb128 0x2
	.long	.LASF76
	.byte	0x3
	.byte	0x29
	.long	0x4b7
	.uleb128 0x19
	.long	0x4a0
	.long	0x4c3
	.uleb128 0x1a
	.byte	0x3
	.byte	0
	.uleb128 0x2
	.long	.LASF77
	.byte	0x3
	.byte	0x3a
	.long	0x4ce
	.uleb128 0x19
	.long	0x4a0
	.long	0x4da
	.uleb128 0x1a
	.byte	0x3
	.byte	0
	.uleb128 0x1b
	.long	.LASF104
	.byte	0x3
	.byte	0x44
	.long	0x4ce
	.byte	0x1
	.uleb128 0x1c
	.string	"A"
	.byte	0x1
	.byte	0x10
	.long	0x4f9
	.uleb128 0x9
	.byte	0x3
	.quad	A
	.uleb128 0x6
	.byte	0x8
	.long	0x4a0
	.uleb128 0x1c
	.string	"B"
	.byte	0x1
	.byte	0x11
	.long	0x4f9
	.uleb128 0x9
	.byte	0x3
	.quad	B
	.uleb128 0x1c
	.string	"C"
	.byte	0x1
	.byte	0x12
	.long	0x4f9
	.uleb128 0x9
	.byte	0x3
	.quad	C
	.uleb128 0x1d
	.long	.LASF105
	.byte	0x1
	.byte	0x4f
	.long	0x5e
	.quad	.LFB4795
	.quad	.LFE4795-.LFB4795
	.uleb128 0x1
	.byte	0x9c
	.long	0x7e9
	.uleb128 0x1e
	.long	.LASF78
	.byte	0x1
	.byte	0x4f
	.long	0x5e
	.long	.LLST20
	.uleb128 0x1e
	.long	.LASF79
	.byte	0x1
	.byte	0x4f
	.long	0x7e9
	.long	.LLST21
	.uleb128 0x1f
	.string	"t0"
	.byte	0x1
	.byte	0x5f
	.long	0x316
	.uleb128 0x3
	.byte	0x91
	.sleb128 -80
	.uleb128 0x1f
	.string	"t1"
	.byte	0x1
	.byte	0x5f
	.long	0x316
	.uleb128 0x2
	.byte	0x91
	.sleb128 -64
	.uleb128 0x20
	.long	.LASF80
	.byte	0x1
	.byte	0x60
	.long	0x499
	.uleb128 0x20
	.long	.LASF81
	.byte	0x1
	.byte	0x60
	.long	0x499
	.uleb128 0x20
	.long	.LASF82
	.byte	0x1
	.byte	0x60
	.long	0x499
	.uleb128 0x21
	.quad	.LBB92
	.quad	.LBE92-.LBB92
	.long	0x5cc
	.uleb128 0x22
	.string	"i"
	.byte	0x1
	.byte	0x55
	.long	0x5e
	.uleb128 0x23
	.quad	.LVL42
	.long	0xc16
	.byte	0
	.uleb128 0x21
	.quad	.LBB93
	.quad	.LBE93-.LBB93
	.long	0x5f8
	.uleb128 0x22
	.string	"i"
	.byte	0x1
	.byte	0x5a
	.long	0x5e
	.uleb128 0x23
	.quad	.LVL43
	.long	0xc16
	.byte	0
	.uleb128 0x24
	.long	0xbf9
	.quad	.LBB94
	.long	.Ldebug_ranges0+0x2d0
	.byte	0x1
	.byte	0x69
	.long	0x639
	.uleb128 0x25
	.long	0xc09
	.long	.LLST22
	.uleb128 0x26
	.quad	.LVL52
	.long	0xc22
	.uleb128 0x27
	.uleb128 0x1
	.byte	0x55
	.uleb128 0x1
	.byte	0x31
	.uleb128 0x27
	.uleb128 0x1
	.byte	0x54
	.uleb128 0x9
	.byte	0x3
	.quad	.LC4
	.byte	0
	.byte	0
	.uleb128 0x24
	.long	0xbf9
	.quad	.LBB102
	.long	.Ldebug_ranges0+0x320
	.byte	0x1
	.byte	0x72
	.long	0x67a
	.uleb128 0x25
	.long	0xc09
	.long	.LLST23
	.uleb128 0x26
	.quad	.LVL60
	.long	0xc22
	.uleb128 0x27
	.uleb128 0x1
	.byte	0x55
	.uleb128 0x1
	.byte	0x31
	.uleb128 0x27
	.uleb128 0x1
	.byte	0x54
	.uleb128 0x9
	.byte	0x3
	.quad	.LC5
	.byte	0
	.byte	0
	.uleb128 0x24
	.long	0xbf9
	.quad	.LBB110
	.long	.Ldebug_ranges0+0x370
	.byte	0x1
	.byte	0x7b
	.long	0x6bb
	.uleb128 0x25
	.long	0xc09
	.long	.LLST24
	.uleb128 0x26
	.quad	.LVL68
	.long	0xc22
	.uleb128 0x27
	.uleb128 0x1
	.byte	0x55
	.uleb128 0x1
	.byte	0x31
	.uleb128 0x27
	.uleb128 0x1
	.byte	0x54
	.uleb128 0x9
	.byte	0x3
	.quad	.LC6
	.byte	0
	.byte	0
	.uleb128 0x28
	.quad	.LVL38
	.long	0xc2d
	.long	0x6d4
	.uleb128 0x27
	.uleb128 0x1
	.byte	0x55
	.uleb128 0x3
	.byte	0x40
	.byte	0x43
	.byte	0x24
	.byte	0
	.uleb128 0x28
	.quad	.LVL39
	.long	0xc2d
	.long	0x6ed
	.uleb128 0x27
	.uleb128 0x1
	.byte	0x55
	.uleb128 0x3
	.byte	0x40
	.byte	0x43
	.byte	0x24
	.byte	0
	.uleb128 0x28
	.quad	.LVL40
	.long	0xc2d
	.long	0x706
	.uleb128 0x27
	.uleb128 0x1
	.byte	0x55
	.uleb128 0x3
	.byte	0x40
	.byte	0x43
	.byte	0x24
	.byte	0
	.uleb128 0x28
	.quad	.LVL45
	.long	0xc39
	.long	0x723
	.uleb128 0x27
	.uleb128 0x1
	.byte	0x55
	.uleb128 0x1
	.byte	0x30
	.uleb128 0x27
	.uleb128 0x1
	.byte	0x54
	.uleb128 0x2
	.byte	0x76
	.sleb128 0
	.byte	0
	.uleb128 0x23
	.quad	.LVL46
	.long	0xadc
	.uleb128 0x28
	.quad	.LVL47
	.long	0xc39
	.long	0x74d
	.uleb128 0x27
	.uleb128 0x1
	.byte	0x55
	.uleb128 0x1
	.byte	0x30
	.uleb128 0x27
	.uleb128 0x1
	.byte	0x54
	.uleb128 0x2
	.byte	0x73
	.sleb128 0
	.byte	0
	.uleb128 0x28
	.quad	.LVL53
	.long	0xc39
	.long	0x76a
	.uleb128 0x27
	.uleb128 0x1
	.byte	0x55
	.uleb128 0x1
	.byte	0x30
	.uleb128 0x27
	.uleb128 0x1
	.byte	0x54
	.uleb128 0x2
	.byte	0x76
	.sleb128 0
	.byte	0
	.uleb128 0x23
	.quad	.LVL54
	.long	0xa47
	.uleb128 0x28
	.quad	.LVL55
	.long	0xc39
	.long	0x794
	.uleb128 0x27
	.uleb128 0x1
	.byte	0x55
	.uleb128 0x1
	.byte	0x30
	.uleb128 0x27
	.uleb128 0x1
	.byte	0x54
	.uleb128 0x2
	.byte	0x73
	.sleb128 0
	.byte	0
	.uleb128 0x28
	.quad	.LVL61
	.long	0xc39
	.long	0x7b1
	.uleb128 0x27
	.uleb128 0x1
	.byte	0x55
	.uleb128 0x1
	.byte	0x30
	.uleb128 0x27
	.uleb128 0x1
	.byte	0x54
	.uleb128 0x2
	.byte	0x76
	.sleb128 0
	.byte	0
	.uleb128 0x23
	.quad	.LVL62
	.long	0x7ef
	.uleb128 0x28
	.quad	.LVL63
	.long	0xc39
	.long	0x7db
	.uleb128 0x27
	.uleb128 0x1
	.byte	0x55
	.uleb128 0x1
	.byte	0x30
	.uleb128 0x27
	.uleb128 0x1
	.byte	0x54
	.uleb128 0x2
	.byte	0x73
	.sleb128 0
	.byte	0
	.uleb128 0x23
	.quad	.LVL69
	.long	0xc44
	.byte	0
	.uleb128 0x6
	.byte	0x8
	.long	0x2bf
	.uleb128 0x29
	.long	.LASF83
	.byte	0x1
	.byte	0x35
	.quad	.LFB4794
	.quad	.LFE4794-.LFB4794
	.uleb128 0x1
	.byte	0x9c
	.long	0xa41
	.uleb128 0x21
	.quad	.LBB61
	.quad	.LBE61-.LBB61
	.long	0x9f7
	.uleb128 0x2a
	.string	"i"
	.byte	0x1
	.byte	0x37
	.long	0x5e
	.long	.LLST11
	.uleb128 0x2b
	.quad	.LBB62
	.quad	.LBE62-.LBB62
	.uleb128 0x2a
	.string	"c"
	.byte	0x1
	.byte	0x38
	.long	0x4f9
	.long	.LLST12
	.uleb128 0x2b
	.quad	.LBB63
	.quad	.LBE63-.LBB63
	.uleb128 0x2a
	.string	"k"
	.byte	0x1
	.byte	0x3d
	.long	0x5e
	.long	.LLST13
	.uleb128 0x2b
	.quad	.LBB64
	.quad	.LBE64-.LBB64
	.uleb128 0x2a
	.string	"b"
	.byte	0x1
	.byte	0x3e
	.long	0xa41
	.long	.LLST14
	.uleb128 0x22
	.string	"a"
	.byte	0x1
	.byte	0x40
	.long	0x4c3
	.uleb128 0x21
	.quad	.LBB67
	.quad	.LBE67-.LBB67
	.long	0x9d2
	.uleb128 0x2a
	.string	"j"
	.byte	0x1
	.byte	0x42
	.long	0x5e
	.long	.LLST16
	.uleb128 0x24
	.long	0xb8b
	.quad	.LBB68
	.long	.Ldebug_ranges0+0x210
	.byte	0x1
	.byte	0x43
	.long	0x8d0
	.uleb128 0x2c
	.long	0xb9c
	.byte	0
	.uleb128 0x24
	.long	0xb8b
	.quad	.LBB71
	.long	.Ldebug_ranges0+0x240
	.byte	0x1
	.byte	0x43
	.long	0x8ed
	.uleb128 0x2c
	.long	0xb9c
	.byte	0
	.uleb128 0x2d
	.long	0xb33
	.quad	.LBB76
	.quad	.LBE76-.LBB76
	.byte	0x1
	.byte	0x43
	.long	0x91c
	.uleb128 0x2c
	.long	0xb59
	.uleb128 0x2c
	.long	0xb4e
	.uleb128 0x25
	.long	0xb43
	.long	.LLST17
	.byte	0
	.uleb128 0x2d
	.long	0xb65
	.quad	.LBB78
	.quad	.LBE78-.LBB78
	.byte	0x1
	.byte	0x43
	.long	0x946
	.uleb128 0x2c
	.long	0xb7e
	.uleb128 0x25
	.long	0xb72
	.long	.LLST18
	.byte	0
	.uleb128 0x24
	.long	0xb8b
	.quad	.LBB80
	.long	.Ldebug_ranges0+0x270
	.byte	0x1
	.byte	0x45
	.long	0x963
	.uleb128 0x2c
	.long	0xb9c
	.byte	0
	.uleb128 0x24
	.long	0xb8b
	.quad	.LBB83
	.long	.Ldebug_ranges0+0x2a0
	.byte	0x1
	.byte	0x45
	.long	0x980
	.uleb128 0x2c
	.long	0xb9c
	.byte	0
	.uleb128 0x2d
	.long	0xb33
	.quad	.LBB88
	.quad	.LBE88-.LBB88
	.byte	0x1
	.byte	0x45
	.long	0x9af
	.uleb128 0x2c
	.long	0xb59
	.uleb128 0x2c
	.long	0xb4e
	.uleb128 0x25
	.long	0xb43
	.long	.LLST19
	.byte	0
	.uleb128 0x2e
	.long	0xb65
	.quad	.LBB90
	.quad	.LBE90-.LBB90
	.byte	0x1
	.byte	0x45
	.uleb128 0x2c
	.long	0xb7e
	.uleb128 0x2c
	.long	0xb72
	.byte	0
	.byte	0
	.uleb128 0x2e
	.long	0xba9
	.quad	.LBB65
	.quad	.LBE65-.LBB65
	.byte	0x1
	.byte	0x40
	.uleb128 0x25
	.long	0xbba
	.long	.LLST15
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.uleb128 0x2f
	.long	0xbc7
	.quad	.LBB57
	.long	.Ldebug_ranges0+0x1e0
	.byte	0x1
	.byte	0x36
	.uleb128 0x25
	.long	0xbed
	.long	.LLST8
	.uleb128 0x25
	.long	0xbe2
	.long	.LLST9
	.uleb128 0x25
	.long	0xbd7
	.long	.LLST10
	.uleb128 0x26
	.quad	.LVL26
	.long	0xc4d
	.uleb128 0x27
	.uleb128 0x1
	.byte	0x54
	.uleb128 0x1
	.byte	0x30
	.uleb128 0x27
	.uleb128 0x1
	.byte	0x51
	.uleb128 0x3
	.byte	0x40
	.byte	0x43
	.byte	0x24
	.byte	0
	.byte	0
	.byte	0
	.uleb128 0x6
	.byte	0x8
	.long	0x4a7
	.uleb128 0x29
	.long	.LASF84
	.byte	0x1
	.byte	0x22
	.quad	.LFB4793
	.quad	.LFE4793-.LFB4793
	.uleb128 0x1
	.byte	0x9c
	.long	0xadc
	.uleb128 0x30
	.long	.Ldebug_ranges0+0x90
	.uleb128 0x2a
	.string	"i"
	.byte	0x1
	.byte	0x23
	.long	0x5e
	.long	.LLST3
	.uleb128 0x30
	.long	.Ldebug_ranges0+0xc0
	.uleb128 0x2a
	.string	"c"
	.byte	0x1
	.byte	0x24
	.long	0x4f9
	.long	.LLST4
	.uleb128 0x31
	.long	0xa97
	.uleb128 0x22
	.string	"j"
	.byte	0x1
	.byte	0x26
	.long	0x5e
	.byte	0
	.uleb128 0x30
	.long	.Ldebug_ranges0+0xf0
	.uleb128 0x22
	.string	"k"
	.byte	0x1
	.byte	0x29
	.long	0x5e
	.uleb128 0x30
	.long	.Ldebug_ranges0+0x140
	.uleb128 0x2a
	.string	"b"
	.byte	0x1
	.byte	0x2a
	.long	0xa41
	.long	.LLST5
	.uleb128 0x2a
	.string	"a"
	.byte	0x1
	.byte	0x2b
	.long	0x4a0
	.long	.LLST6
	.uleb128 0x30
	.long	.Ldebug_ranges0+0x190
	.uleb128 0x2a
	.string	"j"
	.byte	0x1
	.byte	0x2d
	.long	0x5e
	.long	.LLST7
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.uleb128 0x29
	.long	.LASF85
	.byte	0x1
	.byte	0x14
	.quad	.LFB4792
	.quad	.LFE4792-.LFB4792
	.uleb128 0x1
	.byte	0x9c
	.long	0xb33
	.uleb128 0x30
	.long	.Ldebug_ranges0+0
	.uleb128 0x2a
	.string	"i"
	.byte	0x1
	.byte	0x15
	.long	0x5e
	.long	.LLST0
	.uleb128 0x30
	.long	.Ldebug_ranges0+0x30
	.uleb128 0x2a
	.string	"j"
	.byte	0x1
	.byte	0x17
	.long	0x5e
	.long	.LLST1
	.uleb128 0x30
	.long	.Ldebug_ranges0+0x60
	.uleb128 0x2a
	.string	"k"
	.byte	0x1
	.byte	0x1a
	.long	0x5e
	.long	.LLST2
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.uleb128 0x32
	.long	.LASF86
	.byte	0x4
	.byte	0x2f
	.long	0x4c3
	.byte	0x3
	.long	0xb65
	.uleb128 0x33
	.string	"__A"
	.byte	0x4
	.byte	0x2f
	.long	0x4c3
	.uleb128 0x33
	.string	"__B"
	.byte	0x4
	.byte	0x2f
	.long	0x4c3
	.uleb128 0x33
	.string	"__C"
	.byte	0x4
	.byte	0x2f
	.long	0x4c3
	.byte	0
	.uleb128 0x34
	.long	.LASF106
	.byte	0x3
	.value	0x379
	.byte	0x3
	.long	0xb8b
	.uleb128 0x35
	.string	"__P"
	.byte	0x3
	.value	0x379
	.long	0x4f9
	.uleb128 0x35
	.string	"__A"
	.byte	0x3
	.value	0x379
	.long	0x4c3
	.byte	0
	.uleb128 0x36
	.long	.LASF87
	.byte	0x3
	.value	0x373
	.long	0x4c3
	.byte	0x3
	.long	0xba9
	.uleb128 0x35
	.string	"__P"
	.byte	0x3
	.value	0x373
	.long	0xa41
	.byte	0
	.uleb128 0x36
	.long	.LASF88
	.byte	0x3
	.value	0x2d6
	.long	0x4c3
	.byte	0x3
	.long	0xbc7
	.uleb128 0x35
	.string	"__X"
	.byte	0x3
	.value	0x2d6
	.long	0xa41
	.byte	0
	.uleb128 0x32
	.long	.LASF89
	.byte	0x2
	.byte	0x3b
	.long	0x8d
	.byte	0x3
	.long	0xbf9
	.uleb128 0x37
	.long	.LASF90
	.byte	0x2
	.byte	0x3b
	.long	0x8d
	.uleb128 0x37
	.long	.LASF91
	.byte	0x2
	.byte	0x3b
	.long	0x5e
	.uleb128 0x37
	.long	.LASF92
	.byte	0x2
	.byte	0x3b
	.long	0x29
	.byte	0
	.uleb128 0x32
	.long	.LASF93
	.byte	0x5
	.byte	0x66
	.long	0x5e
	.byte	0x3
	.long	0xc16
	.uleb128 0x37
	.long	.LASF94
	.byte	0x5
	.byte	0x66
	.long	0x2ca
	.uleb128 0x38
	.byte	0
	.uleb128 0x39
	.long	.LASF95
	.long	.LASF95
	.byte	0xe
	.value	0x1d3
	.uleb128 0x3a
	.long	.LASF96
	.long	.LASF96
	.byte	0x5
	.byte	0x57
	.uleb128 0x39
	.long	.LASF97
	.long	.LASF97
	.byte	0xe
	.value	0x21b
	.uleb128 0x3a
	.long	.LASF98
	.long	.LASF98
	.byte	0xc
	.byte	0xdb
	.uleb128 0x3b
	.long	.LASF107
	.long	.LASF107
	.uleb128 0x3c
	.long	.LASF89
	.long	.LASF108
	.byte	0xf
	.byte	0
	.long	.LASF89
	.byte	0
	.section	.debug_abbrev,"",@progbits
.Ldebug_abbrev0:
	.uleb128 0x1
	.uleb128 0x11
	.byte	0x1
	.uleb128 0x25
	.uleb128 0xe
	.uleb128 0x13
	.uleb128 0xb
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x1b
	.uleb128 0xe
	.uleb128 0x55
	.uleb128 0x17
	.uleb128 0x11
	.uleb128 0x1
	.uleb128 0x10
	.uleb128 0x17
	.byte	0
	.byte	0
	.uleb128 0x2
	.uleb128 0x16
	.byte	0
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x49
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x3
	.uleb128 0x24
	.byte	0
	.uleb128 0xb
	.uleb128 0xb
	.uleb128 0x3e
	.uleb128 0xb
	.uleb128 0x3
	.uleb128 0xe
	.byte	0
	.byte	0
	.uleb128 0x4
	.uleb128 0x24
	.byte	0
	.uleb128 0xb
	.uleb128 0xb
	.uleb128 0x3e
	.uleb128 0xb
	.uleb128 0x3
	.uleb128 0x8
	.byte	0
	.byte	0
	.uleb128 0x5
	.uleb128 0xf
	.byte	0
	.uleb128 0xb
	.uleb128 0xb
	.byte	0
	.byte	0
	.uleb128 0x6
	.uleb128 0xf
	.byte	0
	.uleb128 0xb
	.uleb128 0xb
	.uleb128 0x49
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x7
	.uleb128 0x26
	.byte	0
	.uleb128 0x49
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x8
	.uleb128 0x13
	.byte	0x1
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0xb
	.uleb128 0xb
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x1
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x9
	.uleb128 0xd
	.byte	0
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x38
	.uleb128 0xb
	.byte	0
	.byte	0
	.uleb128 0xa
	.uleb128 0xd
	.byte	0
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0x5
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x38
	.uleb128 0xb
	.byte	0
	.byte	0
	.uleb128 0xb
	.uleb128 0x16
	.byte	0
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.byte	0
	.byte	0
	.uleb128 0xc
	.uleb128 0x1
	.byte	0x1
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x1
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0xd
	.uleb128 0x21
	.byte	0
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x2f
	.uleb128 0xb
	.byte	0
	.byte	0
	.uleb128 0xe
	.uleb128 0x13
	.byte	0
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3c
	.uleb128 0x19
	.byte	0
	.byte	0
	.uleb128 0xf
	.uleb128 0x34
	.byte	0
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0x5
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x3f
	.uleb128 0x19
	.uleb128 0x3c
	.uleb128 0x19
	.byte	0
	.byte	0
	.uleb128 0x10
	.uleb128 0x37
	.byte	0
	.uleb128 0x49
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x11
	.uleb128 0x34
	.byte	0
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x3f
	.uleb128 0x19
	.uleb128 0x3c
	.uleb128 0x19
	.byte	0
	.byte	0
	.uleb128 0x12
	.uleb128 0x21
	.byte	0
	.byte	0
	.byte	0
	.uleb128 0x13
	.uleb128 0x15
	.byte	0x1
	.uleb128 0x27
	.uleb128 0x19
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x1
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x14
	.uleb128 0x5
	.byte	0
	.uleb128 0x49
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x15
	.uleb128 0x15
	.byte	0x1
	.uleb128 0x27
	.uleb128 0x19
	.uleb128 0x1
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x16
	.uleb128 0x26
	.byte	0
	.byte	0
	.byte	0
	.uleb128 0x17
	.uleb128 0x35
	.byte	0
	.uleb128 0x49
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x18
	.uleb128 0x15
	.byte	0
	.uleb128 0x27
	.uleb128 0x19
	.byte	0
	.byte	0
	.uleb128 0x19
	.uleb128 0x1
	.byte	0x1
	.uleb128 0x2107
	.uleb128 0x19
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x1
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x1a
	.uleb128 0x21
	.byte	0
	.uleb128 0x2f
	.uleb128 0xb
	.byte	0
	.byte	0
	.uleb128 0x1b
	.uleb128 0x16
	.byte	0
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x88
	.uleb128 0xb
	.byte	0
	.byte	0
	.uleb128 0x1c
	.uleb128 0x34
	.byte	0
	.uleb128 0x3
	.uleb128 0x8
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x3f
	.uleb128 0x19
	.uleb128 0x2
	.uleb128 0x18
	.byte	0
	.byte	0
	.uleb128 0x1d
	.uleb128 0x2e
	.byte	0x1
	.uleb128 0x3f
	.uleb128 0x19
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x27
	.uleb128 0x19
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x11
	.uleb128 0x1
	.uleb128 0x12
	.uleb128 0x7
	.uleb128 0x40
	.uleb128 0x18
	.uleb128 0x2117
	.uleb128 0x19
	.uleb128 0x1
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x1e
	.uleb128 0x5
	.byte	0
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x2
	.uleb128 0x17
	.byte	0
	.byte	0
	.uleb128 0x1f
	.uleb128 0x34
	.byte	0
	.uleb128 0x3
	.uleb128 0x8
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x2
	.uleb128 0x18
	.byte	0
	.byte	0
	.uleb128 0x20
	.uleb128 0x34
	.byte	0
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x49
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x21
	.uleb128 0xb
	.byte	0x1
	.uleb128 0x11
	.uleb128 0x1
	.uleb128 0x12
	.uleb128 0x7
	.uleb128 0x1
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x22
	.uleb128 0x34
	.byte	0
	.uleb128 0x3
	.uleb128 0x8
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x49
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x23
	.uleb128 0x4109
	.byte	0
	.uleb128 0x11
	.uleb128 0x1
	.uleb128 0x31
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x24
	.uleb128 0x1d
	.byte	0x1
	.uleb128 0x31
	.uleb128 0x13
	.uleb128 0x52
	.uleb128 0x1
	.uleb128 0x55
	.uleb128 0x17
	.uleb128 0x58
	.uleb128 0xb
	.uleb128 0x59
	.uleb128 0xb
	.uleb128 0x1
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x25
	.uleb128 0x5
	.byte	0
	.uleb128 0x31
	.uleb128 0x13
	.uleb128 0x2
	.uleb128 0x17
	.byte	0
	.byte	0
	.uleb128 0x26
	.uleb128 0x4109
	.byte	0x1
	.uleb128 0x11
	.uleb128 0x1
	.uleb128 0x31
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x27
	.uleb128 0x410a
	.byte	0
	.uleb128 0x2
	.uleb128 0x18
	.uleb128 0x2111
	.uleb128 0x18
	.byte	0
	.byte	0
	.uleb128 0x28
	.uleb128 0x4109
	.byte	0x1
	.uleb128 0x11
	.uleb128 0x1
	.uleb128 0x31
	.uleb128 0x13
	.uleb128 0x1
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x29
	.uleb128 0x2e
	.byte	0x1
	.uleb128 0x3f
	.uleb128 0x19
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x27
	.uleb128 0x19
	.uleb128 0x11
	.uleb128 0x1
	.uleb128 0x12
	.uleb128 0x7
	.uleb128 0x40
	.uleb128 0x18
	.uleb128 0x2117
	.uleb128 0x19
	.uleb128 0x1
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x2a
	.uleb128 0x34
	.byte	0
	.uleb128 0x3
	.uleb128 0x8
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x2
	.uleb128 0x17
	.byte	0
	.byte	0
	.uleb128 0x2b
	.uleb128 0xb
	.byte	0x1
	.uleb128 0x11
	.uleb128 0x1
	.uleb128 0x12
	.uleb128 0x7
	.byte	0
	.byte	0
	.uleb128 0x2c
	.uleb128 0x5
	.byte	0
	.uleb128 0x31
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x2d
	.uleb128 0x1d
	.byte	0x1
	.uleb128 0x31
	.uleb128 0x13
	.uleb128 0x11
	.uleb128 0x1
	.uleb128 0x12
	.uleb128 0x7
	.uleb128 0x58
	.uleb128 0xb
	.uleb128 0x59
	.uleb128 0xb
	.uleb128 0x1
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x2e
	.uleb128 0x1d
	.byte	0x1
	.uleb128 0x31
	.uleb128 0x13
	.uleb128 0x11
	.uleb128 0x1
	.uleb128 0x12
	.uleb128 0x7
	.uleb128 0x58
	.uleb128 0xb
	.uleb128 0x59
	.uleb128 0xb
	.byte	0
	.byte	0
	.uleb128 0x2f
	.uleb128 0x1d
	.byte	0x1
	.uleb128 0x31
	.uleb128 0x13
	.uleb128 0x52
	.uleb128 0x1
	.uleb128 0x55
	.uleb128 0x17
	.uleb128 0x58
	.uleb128 0xb
	.uleb128 0x59
	.uleb128 0xb
	.byte	0
	.byte	0
	.uleb128 0x30
	.uleb128 0xb
	.byte	0x1
	.uleb128 0x55
	.uleb128 0x17
	.byte	0
	.byte	0
	.uleb128 0x31
	.uleb128 0xb
	.byte	0x1
	.uleb128 0x1
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x32
	.uleb128 0x2e
	.byte	0x1
	.uleb128 0x3f
	.uleb128 0x19
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x27
	.uleb128 0x19
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x20
	.uleb128 0xb
	.uleb128 0x34
	.uleb128 0x19
	.uleb128 0x1
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x33
	.uleb128 0x5
	.byte	0
	.uleb128 0x3
	.uleb128 0x8
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x49
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x34
	.uleb128 0x2e
	.byte	0x1
	.uleb128 0x3f
	.uleb128 0x19
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0x5
	.uleb128 0x27
	.uleb128 0x19
	.uleb128 0x20
	.uleb128 0xb
	.uleb128 0x34
	.uleb128 0x19
	.uleb128 0x1
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x35
	.uleb128 0x5
	.byte	0
	.uleb128 0x3
	.uleb128 0x8
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0x5
	.uleb128 0x49
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x36
	.uleb128 0x2e
	.byte	0x1
	.uleb128 0x3f
	.uleb128 0x19
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0x5
	.uleb128 0x27
	.uleb128 0x19
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x20
	.uleb128 0xb
	.uleb128 0x34
	.uleb128 0x19
	.uleb128 0x1
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x37
	.uleb128 0x5
	.byte	0
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x49
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x38
	.uleb128 0x18
	.byte	0
	.byte	0
	.byte	0
	.uleb128 0x39
	.uleb128 0x2e
	.byte	0
	.uleb128 0x3f
	.uleb128 0x19
	.uleb128 0x3c
	.uleb128 0x19
	.uleb128 0x6e
	.uleb128 0xe
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0x5
	.byte	0
	.byte	0
	.uleb128 0x3a
	.uleb128 0x2e
	.byte	0
	.uleb128 0x3f
	.uleb128 0x19
	.uleb128 0x3c
	.uleb128 0x19
	.uleb128 0x6e
	.uleb128 0xe
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.byte	0
	.byte	0
	.uleb128 0x3b
	.uleb128 0x2e
	.byte	0
	.uleb128 0x3f
	.uleb128 0x19
	.uleb128 0x3c
	.uleb128 0x19
	.uleb128 0x6e
	.uleb128 0xe
	.uleb128 0x3
	.uleb128 0xe
	.byte	0
	.byte	0
	.uleb128 0x3c
	.uleb128 0x2e
	.byte	0
	.uleb128 0x3f
	.uleb128 0x19
	.uleb128 0x3c
	.uleb128 0x19
	.uleb128 0x6e
	.uleb128 0xe
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x6e
	.uleb128 0xe
	.byte	0
	.byte	0
	.byte	0
	.section	.debug_loc,"",@progbits
.Ldebug_loc0:
.LLST20:
	.quad	.LVL36
	.quad	.LVL37
	.value	0x1
	.byte	0x55
	.quad	.LVL37
	.quad	.LFE4795
	.value	0x4
	.byte	0xf3
	.uleb128 0x1
	.byte	0x55
	.byte	0x9f
	.quad	0
	.quad	0
.LLST21:
	.quad	.LVL36
	.quad	.LVL38-1
	.value	0x1
	.byte	0x54
	.quad	.LVL38-1
	.quad	.LFE4795
	.value	0x4
	.byte	0xf3
	.uleb128 0x1
	.byte	0x54
	.byte	0x9f
	.quad	0
	.quad	0
.LLST22:
	.quad	.LVL47
	.quad	.LVL52
	.value	0xa
	.byte	0x3
	.quad	.LC4
	.byte	0x9f
	.quad	0
	.quad	0
.LLST23:
	.quad	.LVL55
	.quad	.LVL60
	.value	0xa
	.byte	0x3
	.quad	.LC5
	.byte	0x9f
	.quad	0
	.quad	0
.LLST24:
	.quad	.LVL63
	.quad	.LVL68
	.value	0xa
	.byte	0x3
	.quad	.LC6
	.byte	0x9f
	.quad	0
	.quad	0
.LLST11:
	.quad	.LVL26
	.quad	.LVL27
	.value	0x2
	.byte	0x30
	.byte	0x9f
	.quad	0
	.quad	0
.LLST12:
	.quad	.LVL28
	.quad	.LVL29
	.value	0x1
	.byte	0x50
	.quad	.LVL29
	.quad	.LFE4794
	.value	0x5
	.byte	0x72
	.sleb128 -8224
	.byte	0x9f
	.quad	0
	.quad	0
.LLST13:
	.quad	.LVL28
	.quad	.LVL29
	.value	0x2
	.byte	0x30
	.byte	0x9f
	.quad	0
	.quad	0
.LLST14:
	.quad	.LVL29
	.quad	.LVL32
	.value	0xe
	.byte	0x3
	.quad	B
	.byte	0x6
	.byte	0x75
	.sleb128 0
	.byte	0x22
	.byte	0x9f
	.quad	0
	.quad	0
.LLST16:
	.quad	.LVL31
	.quad	.LVL32
	.value	0x2
	.byte	0x30
	.byte	0x9f
	.quad	0
	.quad	0
.LLST17:
	.quad	.LVL32
	.quad	.LVL33
	.value	0x1
	.byte	0x63
	.quad	0
	.quad	0
.LLST18:
	.quad	.LVL33
	.quad	.LVL34
	.value	0x4
	.byte	0x70
	.sleb128 -96
	.byte	0x9f
	.quad	0
	.quad	0
.LLST19:
	.quad	.LVL34
	.quad	.LVL35
	.value	0x1
	.byte	0x63
	.quad	0
	.quad	0
.LLST15:
	.quad	.LVL29
	.quad	.LVL30
	.value	0xe
	.byte	0x3
	.quad	A
	.byte	0x6
	.byte	0x74
	.sleb128 0
	.byte	0x22
	.byte	0x9f
	.quad	.LVL30
	.quad	.LVL31
	.value	0x1
	.byte	0x50
	.quad	0
	.quad	0
.LLST8:
	.quad	.LVL25
	.quad	.LVL26
	.value	0x4
	.byte	0x40
	.byte	0x43
	.byte	0x24
	.byte	0x9f
	.quad	0
	.quad	0
.LLST9:
	.quad	.LVL25
	.quad	.LVL26
	.value	0x2
	.byte	0x30
	.byte	0x9f
	.quad	0
	.quad	0
.LLST10:
	.quad	.LVL25
	.quad	.LVL26-1
	.value	0x9
	.byte	0x3
	.quad	C
	.quad	0
	.quad	0
.LLST3:
	.quad	.LVL5
	.quad	.LVL6
	.value	0x2
	.byte	0x30
	.byte	0x9f
	.quad	0
	.quad	0
.LLST4:
	.quad	.LVL6
	.quad	.LVL19
	.value	0x1
	.byte	0x5d
	.quad	.LVL19
	.quad	.LVL20
	.value	0x4
	.byte	0x7d
	.sleb128 -8192
	.byte	0x9f
	.quad	.LVL21
	.quad	.LFE4793
	.value	0x1
	.byte	0x5d
	.quad	0
	.quad	0
.LLST5:
	.quad	.LVL7
	.quad	.LVL18
	.value	0xe
	.byte	0x3
	.quad	B
	.byte	0x6
	.byte	0x7b
	.sleb128 0
	.byte	0x22
	.byte	0x9f
	.quad	.LVL18
	.quad	.LVL21
	.value	0x12
	.byte	0x3
	.quad	B
	.byte	0x6
	.byte	0x7b
	.sleb128 0
	.byte	0x22
	.byte	0xa
	.value	0x2000
	.byte	0x1c
	.byte	0x9f
	.quad	.LVL21
	.quad	.LFE4793
	.value	0xe
	.byte	0x3
	.quad	B
	.byte	0x6
	.byte	0x7b
	.sleb128 0
	.byte	0x22
	.byte	0x9f
	.quad	0
	.quad	0
.LLST6:
	.quad	.LVL8
	.quad	.LVL16
	.value	0x1
	.byte	0x62
	.quad	.LVL21
	.quad	.LFE4793
	.value	0x1
	.byte	0x62
	.quad	0
	.quad	0
.LLST7:
	.quad	.LVL8
	.quad	.LVL9
	.value	0x2
	.byte	0x30
	.byte	0x9f
	.quad	.LVL9
	.quad	.LVL10
	.value	0x2
	.byte	0x31
	.byte	0x9f
	.quad	.LVL10
	.quad	.LVL11
	.value	0x2
	.byte	0x32
	.byte	0x9f
	.quad	.LVL12
	.quad	.LVL13
	.value	0x1
	.byte	0x5a
	.quad	.LVL13
	.quad	.LVL14
	.value	0x1
	.byte	0x50
	.quad	.LVL14
	.quad	.LVL15
	.value	0x3
	.byte	0x7a
	.sleb128 -1
	.byte	0x9f
	.quad	.LVL21
	.quad	.LVL22
	.value	0x2
	.byte	0x31
	.byte	0x9f
	.quad	.LVL22
	.quad	.LVL23
	.value	0x2
	.byte	0x30
	.byte	0x9f
	.quad	.LVL24
	.quad	.LFE4793
	.value	0x2
	.byte	0x32
	.byte	0x9f
	.quad	0
	.quad	0
.LLST0:
	.quad	.LVL0
	.quad	.LVL1
	.value	0x2
	.byte	0x30
	.byte	0x9f
	.quad	0
	.quad	0
.LLST1:
	.quad	.LVL1
	.quad	.LVL2
	.value	0x2
	.byte	0x30
	.byte	0x9f
	.quad	0
	.quad	0
.LLST2:
	.quad	.LVL3
	.quad	.LVL4
	.value	0x2
	.byte	0x30
	.byte	0x9f
	.quad	0
	.quad	0
	.section	.debug_aranges,"",@progbits
	.long	0x3c
	.value	0x2
	.long	.Ldebug_info0
	.byte	0x8
	.byte	0
	.value	0
	.value	0
	.quad	.Ltext0
	.quad	.Letext0-.Ltext0
	.quad	.LFB4795
	.quad	.LFE4795-.LFB4795
	.quad	0
	.quad	0
	.section	.debug_ranges,"",@progbits
.Ldebug_ranges0:
	.quad	.LBB35
	.quad	.LBE35
	.quad	.LBB40
	.quad	.LBE40
	.quad	0
	.quad	0
	.quad	.LBB36
	.quad	.LBE36
	.quad	.LBB39
	.quad	.LBE39
	.quad	0
	.quad	0
	.quad	.LBB37
	.quad	.LBE37
	.quad	.LBB38
	.quad	.LBE38
	.quad	0
	.quad	0
	.quad	.LBB41
	.quad	.LBE41
	.quad	.LBB56
	.quad	.LBE56
	.quad	0
	.quad	0
	.quad	.LBB42
	.quad	.LBE42
	.quad	.LBB55
	.quad	.LBE55
	.quad	0
	.quad	0
	.quad	.LBB43
	.quad	.LBE43
	.quad	.LBB52
	.quad	.LBE52
	.quad	.LBB53
	.quad	.LBE53
	.quad	.LBB54
	.quad	.LBE54
	.quad	0
	.quad	0
	.quad	.LBB44
	.quad	.LBE44
	.quad	.LBB49
	.quad	.LBE49
	.quad	.LBB50
	.quad	.LBE50
	.quad	.LBB51
	.quad	.LBE51
	.quad	0
	.quad	0
	.quad	.LBB45
	.quad	.LBE45
	.quad	.LBB46
	.quad	.LBE46
	.quad	.LBB47
	.quad	.LBE47
	.quad	.LBB48
	.quad	.LBE48
	.quad	0
	.quad	0
	.quad	.LBB57
	.quad	.LBE57
	.quad	.LBB60
	.quad	.LBE60
	.quad	0
	.quad	0
	.quad	.LBB68
	.quad	.LBE68
	.quad	.LBB74
	.quad	.LBE74
	.quad	0
	.quad	0
	.quad	.LBB71
	.quad	.LBE71
	.quad	.LBB75
	.quad	.LBE75
	.quad	0
	.quad	0
	.quad	.LBB80
	.quad	.LBE80
	.quad	.LBB86
	.quad	.LBE86
	.quad	0
	.quad	0
	.quad	.LBB83
	.quad	.LBE83
	.quad	.LBB87
	.quad	.LBE87
	.quad	0
	.quad	0
	.quad	.LBB94
	.quad	.LBE94
	.quad	.LBB99
	.quad	.LBE99
	.quad	.LBB100
	.quad	.LBE100
	.quad	.LBB101
	.quad	.LBE101
	.quad	0
	.quad	0
	.quad	.LBB102
	.quad	.LBE102
	.quad	.LBB107
	.quad	.LBE107
	.quad	.LBB108
	.quad	.LBE108
	.quad	.LBB109
	.quad	.LBE109
	.quad	0
	.quad	0
	.quad	.LBB110
	.quad	.LBE110
	.quad	.LBB115
	.quad	.LBE115
	.quad	.LBB116
	.quad	.LBE116
	.quad	.LBB117
	.quad	.LBE117
	.quad	0
	.quad	0
	.quad	.Ltext0
	.quad	.Letext0
	.quad	.LFB4795
	.quad	.LFE4795
	.quad	0
	.quad	0
	.section	.debug_line,"",@progbits
.Ldebug_line0:
	.section	.debug_str,"MS",@progbits,1
.LASF93:
	.string	"printf"
.LASF8:
	.string	"__off_t"
.LASF84:
	.string	"mulstr"
.LASF14:
	.string	"_IO_read_ptr"
.LASF97:
	.string	"malloc"
.LASF26:
	.string	"_chain"
.LASF7:
	.string	"size_t"
.LASF32:
	.string	"_shortbuf"
.LASF91:
	.string	"__ch"
.LASF49:
	.string	"_IO_2_1_stderr_"
.LASF20:
	.string	"_IO_buf_base"
.LASF80:
	.string	"tres1"
.LASF81:
	.string	"tres2"
.LASF108:
	.string	"__builtin_memset"
.LASF73:
	.string	"long long unsigned int"
.LASF106:
	.string	"_mm256_storeu_pd"
.LASF82:
	.string	"tres3"
.LASF98:
	.string	"clock_gettime"
.LASF58:
	.string	"__tzname"
.LASF60:
	.string	"__timezone"
.LASF65:
	.string	"long long int"
.LASF4:
	.string	"signed char"
.LASF101:
	.string	"/mnt/c/Users/Nika/Documents/GitHub/790541_AVP"
.LASF27:
	.string	"_fileno"
.LASF15:
	.string	"_IO_read_end"
.LASF99:
	.string	"GNU C11 7.4.0 -mavx -mfma -mtune=generic -march=x86-64 -g -O3 -fstack-protector-strong"
.LASF6:
	.string	"long int"
.LASF13:
	.string	"_flags"
.LASF21:
	.string	"_IO_buf_end"
.LASF50:
	.string	"stdin"
.LASF75:
	.string	"double"
.LASF96:
	.string	"__printf_chk"
.LASF29:
	.string	"_old_offset"
.LASF34:
	.string	"_offset"
.LASF105:
	.string	"main"
.LASF63:
	.string	"timezone"
.LASF100:
	.string	"mulmatrix.c"
.LASF67:
	.string	"__morecore"
.LASF43:
	.string	"_IO_marker"
.LASF68:
	.string	"__free_hook"
.LASF3:
	.string	"unsigned int"
.LASF87:
	.string	"_mm256_loadu_pd"
.LASF88:
	.string	"_mm256_broadcast_sd"
.LASF55:
	.string	"timespec"
.LASF0:
	.string	"long unsigned int"
.LASF103:
	.string	"_IO_FILE_plus"
.LASF18:
	.string	"_IO_write_ptr"
.LASF53:
	.string	"sys_nerr"
.LASF45:
	.string	"_sbuf"
.LASF2:
	.string	"short unsigned int"
.LASF22:
	.string	"_IO_save_base"
.LASF76:
	.string	"__v4df"
.LASF33:
	.string	"_lock"
.LASF5:
	.string	"short int"
.LASF40:
	.string	"_mode"
.LASF51:
	.string	"stdout"
.LASF83:
	.string	"mulvec"
.LASF64:
	.string	"ptrdiff_t"
.LASF47:
	.string	"_IO_2_1_stdin_"
.LASF69:
	.string	"__malloc_hook"
.LASF57:
	.string	"tv_nsec"
.LASF56:
	.string	"tv_sec"
.LASF11:
	.string	"__syscall_slong_t"
.LASF19:
	.string	"_IO_write_end"
.LASF90:
	.string	"__dest"
.LASF102:
	.string	"_IO_lock_t"
.LASF42:
	.string	"_IO_FILE"
.LASF59:
	.string	"__daylight"
.LASF74:
	.string	"float"
.LASF85:
	.string	"mulmat"
.LASF46:
	.string	"_pos"
.LASF95:
	.string	"drand48"
.LASF25:
	.string	"_markers"
.LASF1:
	.string	"unsigned char"
.LASF28:
	.string	"_flags2"
.LASF92:
	.string	"__len"
.LASF71:
	.string	"__memalign_hook"
.LASF86:
	.string	"_mm256_fmadd_pd"
.LASF70:
	.string	"__realloc_hook"
.LASF61:
	.string	"tzname"
.LASF48:
	.string	"_IO_2_1_stdout_"
.LASF107:
	.string	"__stack_chk_fail"
.LASF62:
	.string	"daylight"
.LASF66:
	.string	"long double"
.LASF12:
	.string	"char"
.LASF44:
	.string	"_next"
.LASF9:
	.string	"__off64_t"
.LASF30:
	.string	"_cur_column"
.LASF16:
	.string	"_IO_read_base"
.LASF24:
	.string	"_IO_save_end"
.LASF104:
	.string	"__m256d_u"
.LASF94:
	.string	"__fmt"
.LASF35:
	.string	"__pad1"
.LASF36:
	.string	"__pad2"
.LASF37:
	.string	"__pad3"
.LASF38:
	.string	"__pad4"
.LASF39:
	.string	"__pad5"
.LASF10:
	.string	"__time_t"
.LASF41:
	.string	"_unused2"
.LASF52:
	.string	"stderr"
.LASF89:
	.string	"memset"
.LASF77:
	.string	"__m256d"
.LASF23:
	.string	"_IO_backup_base"
.LASF54:
	.string	"sys_errlist"
.LASF31:
	.string	"_vtable_offset"
.LASF78:
	.string	"argc"
.LASF72:
	.string	"__after_morecore_hook"
.LASF79:
	.string	"argv"
.LASF17:
	.string	"_IO_write_base"
	.ident	"GCC: (Ubuntu 7.4.0-1ubuntu1~18.04.1) 7.4.0"
	.section	.note.GNU-stack,"",@progbits
