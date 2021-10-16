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
	.type	_mm_malloc.constprop.0, @function
_mm_malloc.constprop.0:
.LFB4796:
	.file 1 "/usr/lib/gcc/x86_64-linux-gnu/7/include/mm_malloc.h"
	.loc 1 38 0
	.cfi_startproc
	subq	$24, %rsp	#,
	.cfi_def_cfa_offset 32
# /usr/lib/gcc/x86_64-linux-gnu/7/include/mm_malloc.h:45:   if (posix_memalign (&__ptr, __alignment, __size) == 0)
	.loc 1 45 0
	movl	$8388608, %edx	#,
	movl	$32, %esi	#,
	movq	%rsp, %rdi	#, tmp89
# /usr/lib/gcc/x86_64-linux-gnu/7/include/mm_malloc.h:38: _mm_malloc (size_t __size, size_t __alignment)
	.loc 1 38 0
	movq	%fs:40, %rax	#, tmp101
	movq	%rax, 8(%rsp)	# tmp101, D.29173
	xorl	%eax, %eax	# tmp101
.LVL0:
# /usr/lib/gcc/x86_64-linux-gnu/7/include/mm_malloc.h:45:   if (posix_memalign (&__ptr, __alignment, __size) == 0)
	.loc 1 45 0
	call	posix_memalign@PLT	#
.LVL1:
	testl	%eax, %eax	# _6
	movq	(%rsp), %rdx	# D.29166, <retval>
	je	.L1	#,
	xorl	%edx, %edx	# <retval>
.L1:
# /usr/lib/gcc/x86_64-linux-gnu/7/include/mm_malloc.h:49: }
	.loc 1 49 0
	movq	8(%rsp), %rcx	# D.29173, tmp102
	xorq	%fs:40, %rcx	#, tmp102
	movq	%rdx, %rax	# <retval>,
	jne	.L9	#,
	addq	$24, %rsp	#,
	.cfi_remember_state
	.cfi_def_cfa_offset 8
	ret
.L9:
	.cfi_restore_state
	call	__stack_chk_fail@PLT	#
.LVL2:
	.cfi_endproc
.LFE4796:
	.size	_mm_malloc.constprop.0, .-_mm_malloc.constprop.0
	.p2align 4,,15
	.globl	mulmat
	.type	mulmat, @function
mulmat:
.LFB4792:
	.file 2 "mulmatrix.c"
	.loc 2 22 0
	.cfi_startproc
.LVL3:
.LBB59:
.LBB60:
.LBB61:
# mulmatrix.c:30:                 C1[i*CC+j] += A[i*AC+k]*B[k*BR+j];
	.loc 2 30 0
	movq	B(%rip), %rax	# B, B.3_14
	movq	A(%rip), %rdi	# A, ivtmp.61
	movq	C1(%rip), %r9	# C1, ivtmp.62
	leaq	8388608(%rdi), %r11	#, _87
	leaq	8388608(%rax), %r10	#, ivtmp.53
	leaq	8396800(%rax), %r8	#, _71
.LVL4:
.L11:
	movq	%r10, %rsi	# ivtmp.53, ivtmp.53
.LBE61:
.LBE60:
.LBE59:
# mulmatrix.c:22: void mulmat(void){     //обычное математическое перемножение матриц
	.loc 2 22 0
	movq	%r9, %rcx	# ivtmp.62, ivtmp.51
.LVL5:
	.p2align 4,,10
	.p2align 3
.L13:
.LBB64:
.LBB63:
# mulmatrix.c:27:             C1[i*CC+j] = 0;
	.loc 2 27 0
	vxorpd	%xmm0, %xmm0, %xmm0	# _22
	leaq	-8388608(%rsi), %rax	#, ivtmp.43
	movq	$0x000000000, (%rcx)	#, MEM[base: _74, offset: 0B]
.LVL6:
	movq	%rdi, %rdx	# ivtmp.61, ivtmp.42
.LVL7:
	.p2align 4,,10
	.p2align 3
.L12:
.LBB62:
# mulmatrix.c:30:                 C1[i*CC+j] += A[i*AC+k]*B[k*BR+j];
	.loc 2 30 0 discriminator 3
	vmovsd	(%rdx), %xmm2	# MEM[base: _41, offset: 0B], tmp142
	addq	$8192, %rax	#, ivtmp.43
	addq	$8, %rdx	#, ivtmp.42
	vfmadd231sd	-8192(%rax), %xmm2, %xmm0	# MEM[base: _40, offset: 0B], tmp142, _22
# mulmatrix.c:28:             for (int k = 0; k < AC; k++)
	.loc 2 28 0 discriminator 3
	cmpq	%rsi, %rax	# ivtmp.53, ivtmp.43
# mulmatrix.c:30:                 C1[i*CC+j] += A[i*AC+k]*B[k*BR+j];
	.loc 2 30 0 discriminator 3
	vmovsd	%xmm0, (%rcx)	# _22, MEM[base: _74, offset: 0B]
# mulmatrix.c:28:             for (int k = 0; k < AC; k++)
	.loc 2 28 0 discriminator 3
	jne	.L12	#,
	leaq	8(%rax), %rsi	#, ivtmp.53
	addq	$8, %rcx	#, ivtmp.51
.LBE62:
# mulmatrix.c:25:         for (int j = 0; j < BC; j++)
	.loc 2 25 0 discriminator 2
	cmpq	%rsi, %r8	# ivtmp.53, _71
	jne	.L13	#,
	addq	$8192, %rdi	#, ivtmp.61
	addq	$8192, %r9	#, ivtmp.62
.LBE63:
# mulmatrix.c:23:     for (int i = 0; i < AR; i++)
	.loc 2 23 0 discriminator 2
	cmpq	%r11, %rdi	# _87, ivtmp.61
	jne	.L11	#,
.LBE64:
# mulmatrix.c:34: }
	.loc 2 34 0
	rep ret
	.cfi_endproc
.LFE4792:
	.size	mulmat, .-mulmat
	.p2align 4,,15
	.globl	mulstr
	.type	mulstr, @function
mulstr:
.LFB4793:
	.loc 2 36 0
	.cfi_startproc
.LVL8:
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
.LBB65:
.LBB66:
# mulmatrix.c:38:         double* c = C2 + i*CC; //указатель на 1 элемент i-ой строки С
	.loc 2 38 0
	movq	C2(%rip), %rax	# C2, C2.5_1
.LBB67:
.LBB68:
# mulmatrix.c:45:             double a = A[i*AC + k];//значение k-го элемента в i-ой строке  А
	.loc 2 45 0
	movq	A(%rip), %rbx	# A, A.7_12
# mulmatrix.c:44:             const double* b = B + k*BC;//указатель на 1 элемент строки В
	.loc 2 44 0
	movq	B(%rip), %r15	# B, B.6_8
# mulmatrix.c:45:             double a = A[i*AC + k];//значение k-го элемента в i-ой строке  А
	.loc 2 45 0
	movq	$0, -96(%rbp)	#, %sfp
	movq	$0, -88(%rbp)	#, %sfp
.LBE68:
.LBE67:
# mulmatrix.c:38:         double* c = C2 + i*CC; //указатель на 1 элемент i-ой строки С
	.loc 2 38 0
	movq	%rax, -104(%rbp)	# C2.5_1, %sfp
.LBB76:
.LBB73:
# mulmatrix.c:45:             double a = A[i*AC + k];//значение k-го элемента в i-ой строке  А
	.loc 2 45 0
	movq	%rbx, -112(%rbp)	# A.7_12, %sfp
	movq	%rax, %r13	# C2.5_1, ivtmp.119
	leaq	32(%rax), %rbx	#, ivtmp.121
	addq	$8388608, %rax	#, _228
	movq	%rax, -120(%rbp)	# _228, %sfp
	leaq	8388608(%r15), %rax	#, _178
	movq	%rbx, -80(%rbp)	# ivtmp.121, %sfp
	movq	%rax, -56(%rbp)	# _178, %sfp
.LVL9:
.L27:
	leaq	8(%r13), %rdi	#, tmp218
	movq	%r13, %rcx	# ivtmp.119, c
	xorl	%eax, %eax	# tmp315
	movq	$0, 0(%r13)	#,* ivtmp.119
	movq	$0, 8184(%r13)	#,
	movl	$1024, %ebx	#, niters.69
	andq	$-8, %rdi	#, tmp218
	movq	%r15, %rsi	# B.6_8, ivtmp.110
.LBE73:
.LBE76:
# mulmatrix.c:38:         double* c = C2 + i*CC; //указатель на 1 элемент i-ой строки С
	.loc 2 38 0
	xorl	%r11d, %r11d	# ivtmp.112
	subq	%rdi, %rcx	# tmp218, c
	addl	$8192, %ecx	#, tmp213
	shrl	$3, %ecx	#,
	rep stosq
	movq	%r13, %rax	# ivtmp.119, tmp221
	shrq	$3, %rax	#, tmp221
	negq	%rax	# tmp222
	movq	-96(%rbp), %rcx	# %sfp, ivtmp.123
	movq	-104(%rbp), %r14	# %sfp, C2.5_1
	movl	%eax, %edi	# tmp222, prolog_loop_niters.67
	movq	-112(%rbp), %r12	# %sfp, ivtmp.109
	andl	$3, %edi	#, prolog_loop_niters.67
	subl	%edi, %ebx	# prolog_loop_niters.67, niters.69
	movl	%edi, %eax	# prolog_loop_niters.67, prolog_loop_adjusted_niters.68
	movl	%edi, -60(%rbp)	# prolog_loop_niters.67, %sfp
	movq	-88(%rbp), %rdi	# %sfp, ivtmp.122
	movl	%ebx, %r9d	# niters.69, bnd.70
	movl	%ebx, -64(%rbp)	# niters.69, %sfp
	andl	$-4, %ebx	#, niters_vector_mult_vf.71
	addq	%rcx, %r12	# ivtmp.123, ivtmp.109
	shrl	$2, %r9d	#, bnd.70
	leaq	(%rax,%rdi), %rdx	#, tmp224
	leaq	0(,%rax,8), %rdi	#, _149
	movq	%rdi, -72(%rbp)	# _149, %sfp
	addq	%rcx, %rdi	# ivtmp.123, tmp226
	leaq	(%r14,%rdx,8), %r8	#, vectp.75
	addq	%r14, %rdi	# C2.5_1, vectp.83
	movl	%ebx, %r14d	# niters_vector_mult_vf.71, niters_vector_mult_vf.71
	.p2align 4,,10
	.p2align 3
.L26:
.LVL10:
	leaq	32(%rsi), %rax	#, tmp227
.LBB77:
.LBB74:
# mulmatrix.c:45:             double a = A[i*AC + k];//значение k-го элемента в i-ой строке  А
	.loc 2 45 0
	vmovsd	(%r12), %xmm1	# MEM[base: _192, offset: 0B], a
.LVL11:
	cmpq	%rax, %r13	# tmp227, ivtmp.119
	jnb	.L32	#,
	cmpq	-80(%rbp), %rsi	# %sfp, ivtmp.110
	jb	.L28	#,
.L32:
	movl	-60(%rbp), %eax	# %sfp, prolog_loop_niters.67
	testl	%eax, %eax	# prolog_loop_niters.67
	je	.L29	#,
.LBB69:
# mulmatrix.c:48:             c[j] += a * b[j];//формируем полусуммы
	.loc 2 48 0
	vmovsd	(%rsi), %xmm0	# MEM[base: _183, offset: 0B], tmp232
	cmpl	$1, %eax	#, prolog_loop_niters.67
	vfmadd213sd	0(%r13), %xmm1, %xmm0	# MEM[base: c_35, offset: 0B], a, tmp232
	vmovsd	%xmm0, 0(%r13)	# tmp232, MEM[base: c_35, offset: 0B]
.LVL12:
	je	.L30	#,
	vmovsd	8(%rsi), %xmm0	# MEM[base: _183, offset: 8B], tmp233
	cmpl	$3, %eax	#, prolog_loop_niters.67
	vfmadd213sd	8(%r13), %xmm1, %xmm0	# MEM[base: c_35, offset: 8B], a, tmp233
	vmovsd	%xmm0, 8(%r13)	# tmp233, MEM[base: c_35, offset: 8B]
.LVL13:
	jne	.L31	#,
	vmovsd	16(%rsi), %xmm0	# MEM[base: _183, offset: 16B], tmp234
	movl	$1021, %ebx	#, ivtmp_80
# mulmatrix.c:47:             for (int j = 0; j < CC; j++){//идём по строке С
	.loc 2 47 0
	movl	$3, %r10d	#, j
# mulmatrix.c:48:             c[j] += a * b[j];//формируем полусуммы
	.loc 2 48 0
	vfmadd213sd	16(%r13), %xmm1, %xmm0	# MEM[base: c_35, offset: 16B], a, tmp234
	vmovsd	%xmm0, 16(%r13)	# tmp234, MEM[base: c_35, offset: 16B]
.LVL14:
.L21:
	movq	-72(%rbp), %rax	# %sfp, _149
	vmovddup	%xmm1, %xmm2	# a, vect_cst__154
# mulmatrix.c:47:             for (int j = 0; j < CC; j++){//идём по строке С
	.loc 2 47 0
	xorl	%edx, %edx	# ivtmp.93
	leaq	(%rax,%r11), %rcx	#, tmp235
	vinsertf128	$1, %xmm2, %ymm2, %ymm2	# vect_cst__154, vect_cst__154
	xorl	%eax, %eax	# ivtmp.96
	addq	%r15, %rcx	# B.6_8, vectp.78
	.p2align 4,,10
	.p2align 3
.L23:
# mulmatrix.c:48:             c[j] += a * b[j];//формируем полусуммы
	.loc 2 48 0 discriminator 3
	vmovupd	(%rcx,%rax), %xmm0	# MEM[base: vectp.78_148, index: ivtmp.96_111, offset: 0B], MEM[base: vectp.78_148, index: ivtmp.96_111, offset: 0B]
	addl	$1, %edx	#, ivtmp.93
	vinsertf128	$0x1, 16(%rcx,%rax), %ymm0, %ymm0	# MEM[base: vectp.78_148, index: ivtmp.96_111, offset: 0B], MEM[base: vectp.78_148, index: ivtmp.96_111, offset: 0B], vect__22.79
	vfmadd213pd	(%r8,%rax), %ymm2, %ymm0	# MEM[base: vectp.75_140, index: ivtmp.96_111, offset: 0B], vect_cst__154, vect__24.81
	vmovapd	%ymm0, (%rdi,%rax)	# vect__24.81, MEM[base: vectp.83_157, index: ivtmp.96_111, offset: 0B]
	addq	$32, %rax	#, ivtmp.96
	cmpl	%r9d, %edx	# bnd.70, ivtmp.93
	jb	.L23	#,
	addl	%r14d, %r10d	# niters_vector_mult_vf.71, tmp.72
	subl	%r14d, %ebx	# niters_vector_mult_vf.71, tmp.73
	cmpl	%r14d, -64(%rbp)	# niters_vector_mult_vf.71, %sfp
	je	.L25	#,
.LVL15:
# mulmatrix.c:48:             c[j] += a * b[j];//формируем полусуммы
	.loc 2 48 0 is_stmt 0
	movslq	%r10d, %rax	# tmp.72, tmp.72
	salq	$3, %rax	#, _17
	leaq	0(%r13,%rax), %rdx	#, _18
	addq	%r15, %rax	# B.6_8, tmp241
# mulmatrix.c:47:             for (int j = 0; j < CC; j++){//идём по строке С
	.loc 2 47 0 is_stmt 1
	cmpl	$1, %ebx	#, tmp.73
# mulmatrix.c:48:             c[j] += a * b[j];//формируем полусуммы
	.loc 2 48 0
	vmovsd	(%rax,%r11), %xmm0	# *_33, tmp243
# mulmatrix.c:47:             for (int j = 0; j < CC; j++){//идём по строке С
	.loc 2 47 0
	leal	1(%r10), %eax	#, j
.LVL16:
# mulmatrix.c:48:             c[j] += a * b[j];//формируем полусуммы
	.loc 2 48 0
	vfmadd213sd	(%rdx), %xmm1, %xmm0	# *_18, a, tmp243
	vmovsd	%xmm0, (%rdx)	# tmp243, *_18
# mulmatrix.c:47:             for (int j = 0; j < CC; j++){//идём по строке С
	.loc 2 47 0
	je	.L25	#,
# mulmatrix.c:48:             c[j] += a * b[j];//формируем полусуммы
	.loc 2 48 0
	cltq
# mulmatrix.c:47:             for (int j = 0; j < CC; j++){//идём по строке С
	.loc 2 47 0
	addl	$2, %r10d	#, j
# mulmatrix.c:48:             c[j] += a * b[j];//формируем полусуммы
	.loc 2 48 0
	salq	$3, %rax	#, _167
.LVL17:
	leaq	0(%r13,%rax), %rdx	#, _168
	addq	%r15, %rax	# B.6_8, tmp246
# mulmatrix.c:47:             for (int j = 0; j < CC; j++){//идём по строке С
	.loc 2 47 0
	cmpl	$2, %ebx	#, tmp.73
# mulmatrix.c:48:             c[j] += a * b[j];//формируем полусуммы
	.loc 2 48 0
	vmovsd	(%rax,%r11), %xmm0	# *_171, tmp248
	vfmadd213sd	(%rdx), %xmm1, %xmm0	# *_168, a, tmp248
	vmovsd	%xmm0, (%rdx)	# tmp248, *_168
.LVL18:
# mulmatrix.c:47:             for (int j = 0; j < CC; j++){//идём по строке С
	.loc 2 47 0
	je	.L25	#,
# mulmatrix.c:48:             c[j] += a * b[j];//формируем полусуммы
	.loc 2 48 0
	movslq	%r10d, %r10	# j, j
	leaq	0(,%r10,8), %rax	#, _122
	leaq	0(%r13,%rax), %rdx	#, _123
	addq	%r15, %rax	# B.6_8, tmp250
	vmovsd	(%rdx), %xmm3	# *_123, tmp342
	vfmadd132sd	(%rax,%r11), %xmm3, %xmm1	# *_126, tmp342, tmp252
.LVL19:
	vmovsd	%xmm1, (%rdx)	# tmp252, *_123
.L25:
.LVL20:
	addq	$8, %r12	#, ivtmp.109
	addq	$8192, %rsi	#, ivtmp.110
	addq	$8192, %r11	#, ivtmp.112
.LVL21:
.LBE69:
.LBE74:
# mulmatrix.c:43:         for (int k = 0; k < AC; k++){//AC=BR, идём по столбцам B и элементам в строке А, k = номер столбца А/номер строки В
	.loc 2 43 0 discriminator 2
	cmpq	%rsi, -56(%rbp)	# ivtmp.110, %sfp
	jne	.L26	#,
	addq	$8192, %r13	#, ivtmp.119
.LVL22:
	addq	$8192, -80(%rbp)	#, %sfp
	addq	$1024, -88(%rbp)	#, %sfp
	addq	$8192, -96(%rbp)	#, %sfp
.LBE77:
.LBE66:
# mulmatrix.c:37:     for (int i = 0; i < AR; i++){//идём по строкам А и С, i = номер строки в А
	.loc 2 37 0 discriminator 2
	cmpq	-120(%rbp), %r13	# %sfp, ivtmp.119
	jne	.L27	#,
	vzeroupper
.LBE65:
# mulmatrix.c:53: }
	.loc 2 53 0
	popq	%rbx	#
	popq	%r10	#
	.cfi_remember_state
	.cfi_def_cfa 10, 0
	popq	%r12	#
	popq	%r13	#
.LVL23:
	popq	%r14	#
	popq	%r15	#
	popq	%rbp	#
	leaq	-8(%r10), %rsp	#,
	.cfi_def_cfa 7, 8
	ret
.LVL24:
	.p2align 4,,10
	.p2align 3
.L30:
	.cfi_restore_state
.LBB80:
.LBB79:
.LBB78:
.LBB75:
.LBB70:
# mulmatrix.c:48:             c[j] += a * b[j];//формируем полусуммы
	.loc 2 48 0
	movl	$1023, %ebx	#, ivtmp_80
# mulmatrix.c:47:             for (int j = 0; j < CC; j++){//идём по строке С
	.loc 2 47 0
	movl	$1, %r10d	#, j
	jmp	.L21	#
.LVL25:
.L29:
.LBE70:
# mulmatrix.c:45:             double a = A[i*AC + k];//значение k-го элемента в i-ой строке  А
	.loc 2 45 0
	movl	$1024, %ebx	#, ivtmp_80
.LBB71:
# mulmatrix.c:47:             for (int j = 0; j < CC; j++){//идём по строке С
	.loc 2 47 0
	xorl	%r10d, %r10d	# j
	jmp	.L21	#
.L28:
.LBE71:
# mulmatrix.c:45:             double a = A[i*AC + k];//значение k-го элемента в i-ой строке  А
	.loc 2 45 0
	xorl	%eax, %eax	# ivtmp.92
.LVL26:
	.p2align 4,,10
	.p2align 3
.L19:
.LBB72:
# mulmatrix.c:48:             c[j] += a * b[j];//формируем полусуммы
	.loc 2 48 0
	vmovsd	(%rsi,%rax), %xmm0	# MEM[base: _183, index: ivtmp.92_112, offset: 0B], tmp253
	vfmadd213sd	0(%r13,%rax), %xmm1, %xmm0	# MEM[base: c_35, index: ivtmp.92_112, offset: 0B], a, tmp253
	vmovsd	%xmm0, 0(%r13,%rax)	# tmp253, MEM[base: c_35, index: ivtmp.92_112, offset: 0B]
	addq	$8, %rax	#, ivtmp.92
# mulmatrix.c:47:             for (int j = 0; j < CC; j++){//идём по строке С
	.loc 2 47 0
	cmpq	$8192, %rax	#, ivtmp.92
	jne	.L19	#,
	jmp	.L25	#
.LVL27:
.L31:
# mulmatrix.c:48:             c[j] += a * b[j];//формируем полусуммы
	.loc 2 48 0
	movl	$1022, %ebx	#, ivtmp_80
# mulmatrix.c:47:             for (int j = 0; j < CC; j++){//идём по строке С
	.loc 2 47 0
	movl	$2, %r10d	#, j
	jmp	.L21	#
.LBE72:
.LBE75:
.LBE78:
.LBE79:
.LBE80:
	.cfi_endproc
.LFE4793:
	.size	mulstr, .-mulstr
	.p2align 4,,15
	.globl	mulvec
	.type	mulvec, @function
mulvec:
.LFB4794:
	.loc 2 55 0
	.cfi_startproc
.LVL28:
	leaq	8(%rsp), %r10	#,
	.cfi_def_cfa 10, 0
	andq	$-32, %rsp	#,
.LBB81:
.LBB82:
# /usr/include/x86_64-linux-gnu/bits/string_fortified.h:71:   return __builtin___memset_chk (__dest, __ch, __len, __bos0 (__dest));
	.file 3 "/usr/include/x86_64-linux-gnu/bits/string_fortified.h"
	.loc 3 71 0
	movl	$8388608, %edx	#,
	xorl	%esi, %esi	#
.LBE82:
.LBE81:
# mulmatrix.c:55: void mulvec(void){      //векторная оптимизация
	.loc 2 55 0
	pushq	-8(%r10)	#
	pushq	%rbp	#
	.cfi_escape 0x10,0x6,0x2,0x76,0
	movq	%rsp, %rbp	#,
	pushq	%r10	#
	.cfi_escape 0xf,0x3,0x76,0x78,0x6
	subq	$8, %rsp	#,
.LBB84:
.LBB83:
# /usr/include/x86_64-linux-gnu/bits/string_fortified.h:71:   return __builtin___memset_chk (__dest, __ch, __len, __bos0 (__dest));
	.loc 3 71 0
	movq	C3(%rip), %rdi	# C3,
	call	memset@PLT	#
.LVL29:
	xorl	%r9d, %r9d	# ivtmp.156
.LVL30:
.L47:
.LBE83:
.LBE84:
.LBB85:
.LBB86:
# mulmatrix.c:59:         double* ap = A + i*AC;
	.loc 2 59 0
	movq	A(%rip), %rsi	# A, ap
# mulmatrix.c:58:         double* c = C3 + i*CC; //указатель на 1 элемент i-ой строки С
	.loc 2 58 0
	movq	C3(%rip), %r8	# C3, c
# mulmatrix.c:60:         const double* b = B;//указатель на 1 элемент матрицы В
	.loc 2 60 0
	movq	B(%rip), %rdx	# B, b
# mulmatrix.c:59:         double* ap = A + i*AC;
	.loc 2 59 0
	addq	%r9, %rsi	# ivtmp.156, ap
# mulmatrix.c:58:         double* c = C3 + i*CC; //указатель на 1 элемент i-ой строки С
	.loc 2 58 0
	addq	%r9, %r8	# ivtmp.156, c
.LVL31:
	leaq	8192(%rsi), %rdi	#, _13
.LVL32:
	.p2align 4,,10
	.p2align 3
.L46:
.LBB87:
.LBB88:
.LBB89:
.LBB90:
# /usr/lib/gcc/x86_64-linux-gnu/7/include/avxintrin.h:728:   return (__m256d) __builtin_ia32_vbroadcastsd256 (__X);
	.file 4 "/usr/lib/gcc/x86_64-linux-gnu/7/include/avxintrin.h"
	.loc 4 728 0
	vbroadcastsd	(%rsi), %ymm0	#* ap, _28
.LVL33:
	leaq	8192(%rdx), %rcx	#, b
.LBE90:
.LBE89:
# mulmatrix.c:65:             double* cp = c;
	.loc 2 65 0
	movq	%r8, %rax	# c, cp
.LVL34:
	.p2align 4,,10
	.p2align 3
.L45:
.LBB91:
.LBB92:
.LBB93:
# /usr/lib/gcc/x86_64-linux-gnu/7/include/fmaintrin.h:49:   return (__m256d)__builtin_ia32_vfmaddpd256 ((__v4df)__A, (__v4df)__B,
	.file 5 "/usr/lib/gcc/x86_64-linux-gnu/7/include/fmaintrin.h"
	.loc 5 49 0
	vmovapd	(%rdx), %ymm1	# MEM[base: b_57, offset: 0B], tmp125
	addq	$64, %rdx	#, b
.LVL35:
	addq	$64, %rax	#, cp
.LVL36:
	vfmadd213pd	-64(%rax), %ymm0, %ymm1	# MEM[base: cp_60, offset: 0B], _28, tmp125
.LVL37:
.LBE93:
.LBE92:
.LBB94:
.LBB95:
# /usr/lib/gcc/x86_64-linux-gnu/7/include/avxintrin.h:867:   *(__m256d *)__P = __A;
	.loc 4 867 0
	vmovapd	%ymm1, -64(%rax)	# tmp125, MEM[base: cp_60, offset: 0B]
.LVL38:
.LBE95:
.LBE94:
.LBB96:
.LBB97:
# /usr/lib/gcc/x86_64-linux-gnu/7/include/fmaintrin.h:49:   return (__m256d)__builtin_ia32_vfmaddpd256 ((__v4df)__A, (__v4df)__B,
	.loc 5 49 0
	vmovapd	-32(%rdx), %ymm1	# MEM[base: b_57, offset: 32B], tmp128
	vfmadd213pd	-32(%rax), %ymm0, %ymm1	# MEM[base: cp_60, offset: 32B], _28, tmp128
.LVL39:
.LBE97:
.LBE96:
.LBB98:
.LBB99:
# /usr/lib/gcc/x86_64-linux-gnu/7/include/avxintrin.h:867:   *(__m256d *)__P = __A;
	.loc 4 867 0
	vmovapd	%ymm1, -32(%rax)	# tmp128, MEM[base: cp_60, offset: 32B]
.LVL40:
.LBE99:
.LBE98:
# mulmatrix.c:66:             for (int j = CC>>3; j > 0; j--){//идём по строке С
	.loc 2 66 0
	cmpq	%rdx, %rcx	# b, b
	jne	.L45	#,
.LBE91:
# mulmatrix.c:77:             ap++;
	.loc 2 77 0 discriminator 2
	addq	$8, %rsi	#, ap
.LVL41:
	movq	%rcx, %rdx	# b, b
.LBE88:
# mulmatrix.c:62:         for (int k = AC; k > 0; k--){//AC=BR, идём по столбцам B и элементам в строке А, k = номер столбца А/номер строки В
	.loc 2 62 0 discriminator 2
	cmpq	%rsi, %rdi	# ap, _13
	jne	.L46	#,
.LVL42:
	addq	$8192, %r9	#, ivtmp.156
.LBE87:
.LBE86:
# mulmatrix.c:57:     for (int i = 0; i < AR; i++){//идём по строкам А и С, i = номер строки в А
	.loc 2 57 0 discriminator 2
	cmpq	$8388608, %r9	#, ivtmp.156
	jne	.L47	#,
	vzeroupper
.LBE85:
# mulmatrix.c:82: }
	.loc 2 82 0
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
	.string	"\320\222\321\200\320\265\320\274\321\217 \321\200\320\260\321\201\321\201\321\207\321\221\321\202\320\260 \320\274\320\260\321\202\321\200\320\270\321\206\321\213 \320\241\320\242\320\240\320\236\320\247\320\235\320\253\320\234 \321\201\320\277\320\276\321\201\320\276\320\261\320\276\320\274: %2.5f mksec, %.4f  \n"
	.align 8
.LC6:
	.string	"\320\222\321\200\320\265\320\274\321\217 \321\200\320\260\321\201\321\201\321\207\321\221\321\202\320\260 \320\274\320\260\321\202\321\200\320\270\321\206\321\213 \320\222\320\225\320\232\320\242\320\236\320\240\320\235\320\253\320\234 \321\201\320\277\320\276\321\201\320\276\320\261\320\276\320\274: %2.5f mksec, %.4f \n"
	.align 8
.LC7:
	.string	"\320\240\320\265\320\267\321\203\320\273\321\214\321\202\320\270\321\200\321\203\321\216\321\211\320\270\320\265 \320\274\320\260\321\202\321\200\320\270\321\206\321\213 \320\235\320\225 \321\200\320\260\320\262\320\275\321\213!"
	.section	.text.startup,"ax",@progbits
	.p2align 4,,15
	.globl	main
	.type	main, @function
main:
.LFB4795:
	.loc 2 85 0
	.cfi_startproc
.LVL43:
	pushq	%rbp	#
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	pushq	%rbx	#
	.cfi_def_cfa_offset 24
	.cfi_offset 3, -24
# mulmatrix.c:90:     C3 = _mm_malloc(sizeof(double)*CR*CC,  4*(sizeof(double)));
	.loc 2 90 0
	xorl	%ebx, %ebx	# ivtmp.181
# mulmatrix.c:85: {
	.loc 2 85 0
	subq	$72, %rsp	#,
	.cfi_def_cfa_offset 96
# mulmatrix.c:85: {
	.loc 2 85 0
	movq	%fs:40, %rax	#, tmp205
	movq	%rax, 56(%rsp)	# tmp205, D.29326
	xorl	%eax, %eax	# tmp205
# mulmatrix.c:86:     A = _mm_malloc(sizeof(double)*AR*AC, 4*(sizeof(double)));
	.loc 2 86 0
	call	_mm_malloc.constprop.0	#
.LVL44:
	movq	%rax, A(%rip)	# _1, A
# mulmatrix.c:87:     B = _mm_malloc(sizeof(double)*BR*BC,  4*(sizeof(double)));
	.loc 2 87 0
	call	_mm_malloc.constprop.0	#
.LVL45:
	movq	%rax, B(%rip)	# _2, B
# mulmatrix.c:88:     C1 = _mm_malloc(sizeof(double)*CR*CC,  4*(sizeof(double)));
	.loc 2 88 0
	call	_mm_malloc.constprop.0	#
.LVL46:
	movq	%rax, C1(%rip)	# _3, C1
# mulmatrix.c:89:     C2 = _mm_malloc(sizeof(double)*CR*CC,  4*(sizeof(double)));
	.loc 2 89 0
	call	_mm_malloc.constprop.0	#
.LVL47:
	movq	%rax, C2(%rip)	# _4, C2
# mulmatrix.c:90:     C3 = _mm_malloc(sizeof(double)*CR*CC,  4*(sizeof(double)));
	.loc 2 90 0
	call	_mm_malloc.constprop.0	#
.LVL48:
	movq	%rax, C3(%rip)	# _5, C3
.LVL49:
	.p2align 4,,10
	.p2align 3
.L53:
.LBB100:
# mulmatrix.c:94:         A[i] = drand48()*5; 
	.loc 2 94 0 discriminator 3
	call	drand48@PLT	#
.LVL50:
	vmulsd	.LC1(%rip), %xmm0, %xmm0	#, _6, tmp156
	movq	A(%rip), %rax	# A, A
	vmovsd	%xmm0, (%rax,%rbx)	# tmp156, *_10
	addq	$8, %rbx	#, ivtmp.181
# mulmatrix.c:92:     for (int i = 0; i < AC*AR; i++)
	.loc 2 92 0 discriminator 3
	cmpq	$8388608, %rbx	#, ivtmp.181
	jne	.L53	#,
	xorl	%ebx, %ebx	# ivtmp.174
	.p2align 4,,10
	.p2align 3
.L54:
.LBE100:
.LBB101:
# mulmatrix.c:99:         B[i] = drand48()*5; 
	.loc 2 99 0 discriminator 3
	call	drand48@PLT	#
.LVL51:
	vmulsd	.LC1(%rip), %xmm0, %xmm0	#, _12, tmp159
	movq	B(%rip), %rax	# B, B
	vmovsd	%xmm0, (%rax,%rbx)	# tmp159, *_16
.LVL52:
	addq	$8, %rbx	#, ivtmp.174
# mulmatrix.c:97:     for (int i = 0; i <BC*BR; i++)
	.loc 2 97 0 discriminator 3
	cmpq	$8388608, %rbx	#, ivtmp.174
	jne	.L54	#,
.LBE101:
# mulmatrix.c:105:     clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t0);
	.loc 2 105 0
	leaq	16(%rsp), %rbp	#, tmp161
# mulmatrix.c:109:     clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t1);
	.loc 2 109 0
	leaq	32(%rsp), %rbx	#, tmp162
# mulmatrix.c:105:     clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t0);
	.loc 2 105 0
	movl	$3, %edi	#,
	movq	%rbp, %rsi	# tmp161,
	call	clock_gettime@PLT	#
.LVL53:
# mulmatrix.c:109:     clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t1);
	.loc 2 109 0
	movq	%rbx, %rsi	# tmp162,
	movl	$3, %edi	#,
	call	clock_gettime@PLT	#
.LVL54:
# mulmatrix.c:111:     tres1 = ((double)(t1.tv_sec - t0.tv_sec)*10e9 + (double)(t1.tv_nsec - t0.tv_nsec))/10e6;
	.loc 2 111 0
	movq	32(%rsp), %rax	# t1.tv_sec, t1.tv_sec
	subq	16(%rsp), %rax	# t0.tv_sec, tmp163
.LBB102:
.LBB103:
# /usr/include/x86_64-linux-gnu/bits/stdio2.h:104:   return __printf_chk (__USE_FORTIFY_LEVEL - 1, __fmt, __va_arg_pack ());
	.file 6 "/usr/include/x86_64-linux-gnu/bits/stdio2.h"
	.loc 6 104 0
	leaq	.LC4(%rip), %rsi	#,
.LBE103:
.LBE102:
# mulmatrix.c:111:     tres1 = ((double)(t1.tv_sec - t0.tv_sec)*10e9 + (double)(t1.tv_nsec - t0.tv_nsec))/10e6;
	.loc 2 111 0
	vxorpd	%xmm1, %xmm1, %xmm1	# tmp165
.LBB106:
.LBB104:
# /usr/include/x86_64-linux-gnu/bits/stdio2.h:104:   return __printf_chk (__USE_FORTIFY_LEVEL - 1, __fmt, __va_arg_pack ());
	.loc 6 104 0
	movl	$1, %edi	#,
.LBE104:
.LBE106:
# mulmatrix.c:111:     tres1 = ((double)(t1.tv_sec - t0.tv_sec)*10e9 + (double)(t1.tv_nsec - t0.tv_nsec))/10e6;
	.loc 2 111 0
	vxorpd	%xmm0, %xmm0, %xmm0	# tmp168
	vcvtsi2sdq	%rax, %xmm1, %xmm1	# tmp163, tmp165, tmp165
	movq	40(%rsp), %rax	# t1.tv_nsec, t1.tv_nsec
	subq	24(%rsp), %rax	# t0.tv_nsec, tmp166
	vcvtsi2sdq	%rax, %xmm0, %xmm0	# tmp166, tmp168, tmp168
	vfmadd132sd	.LC2(%rip), %xmm0, %xmm1	#, tmp168, tmp169
	vdivsd	.LC3(%rip), %xmm1, %xmm1	#, tmp169, tres1
.LVL55:
.LBB107:
.LBB105:
# /usr/include/x86_64-linux-gnu/bits/stdio2.h:104:   return __printf_chk (__USE_FORTIFY_LEVEL - 1, __fmt, __va_arg_pack ());
	.loc 6 104 0
	movl	$1, %eax	#,
	vmovapd	%xmm1, %xmm0	# tres1,
	vmovsd	%xmm1, 8(%rsp)	# tres1, %sfp
	call	__printf_chk@PLT	#
.LVL56:
.LBE105:
.LBE107:
# mulmatrix.c:114:     clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t0);
	.loc 2 114 0
	movq	%rbp, %rsi	# tmp161,
	movl	$3, %edi	#,
	call	clock_gettime@PLT	#
.LVL57:
# mulmatrix.c:116:     mulstr();
	.loc 2 116 0
	call	mulstr	#
.LVL58:
# mulmatrix.c:118:     clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t1);
	.loc 2 118 0
	movq	%rbx, %rsi	# tmp162,
	movl	$3, %edi	#,
	call	clock_gettime@PLT	#
.LVL59:
# mulmatrix.c:120:     tres2 = ((double)(t1.tv_sec - t0.tv_sec)*10e9 + (double)(t1.tv_nsec - t0.tv_nsec))/10e6;
	.loc 2 120 0
	movq	32(%rsp), %rax	# t1.tv_sec, t1.tv_sec
	subq	16(%rsp), %rax	# t0.tv_sec, tmp174
.LBB108:
.LBB109:
# /usr/include/x86_64-linux-gnu/bits/stdio2.h:104:   return __printf_chk (__USE_FORTIFY_LEVEL - 1, __fmt, __va_arg_pack ());
	.loc 6 104 0
	leaq	.LC5(%rip), %rsi	#,
.LBE109:
.LBE108:
# mulmatrix.c:120:     tres2 = ((double)(t1.tv_sec - t0.tv_sec)*10e9 + (double)(t1.tv_nsec - t0.tv_nsec))/10e6;
	.loc 2 120 0
	vxorpd	%xmm2, %xmm2, %xmm2	# tmp176
.LBB113:
.LBB110:
# /usr/include/x86_64-linux-gnu/bits/stdio2.h:104:   return __printf_chk (__USE_FORTIFY_LEVEL - 1, __fmt, __va_arg_pack ());
	.loc 6 104 0
	movl	$1, %edi	#,
.LBE110:
.LBE113:
# mulmatrix.c:120:     tres2 = ((double)(t1.tv_sec - t0.tv_sec)*10e9 + (double)(t1.tv_nsec - t0.tv_nsec))/10e6;
	.loc 2 120 0
	vxorpd	%xmm0, %xmm0, %xmm0	# tmp179
.LBB114:
.LBB111:
# /usr/include/x86_64-linux-gnu/bits/stdio2.h:104:   return __printf_chk (__USE_FORTIFY_LEVEL - 1, __fmt, __va_arg_pack ());
	.loc 6 104 0
	vmovsd	8(%rsp), %xmm1	# %sfp, tres1
.LBE111:
.LBE114:
# mulmatrix.c:120:     tres2 = ((double)(t1.tv_sec - t0.tv_sec)*10e9 + (double)(t1.tv_nsec - t0.tv_nsec))/10e6;
	.loc 2 120 0
	vcvtsi2sdq	%rax, %xmm2, %xmm2	# tmp174, tmp176, tmp176
	movq	40(%rsp), %rax	# t1.tv_nsec, t1.tv_nsec
	subq	24(%rsp), %rax	# t0.tv_nsec, tmp177
	vcvtsi2sdq	%rax, %xmm0, %xmm0	# tmp177, tmp179, tmp179
	vfmadd132sd	.LC2(%rip), %xmm0, %xmm2	#, tmp179, tmp180
	vdivsd	.LC3(%rip), %xmm2, %xmm2	#, tmp180, tres2
.LVL60:
.LBB115:
.LBB112:
# /usr/include/x86_64-linux-gnu/bits/stdio2.h:104:   return __printf_chk (__USE_FORTIFY_LEVEL - 1, __fmt, __va_arg_pack ());
	.loc 6 104 0
	movl	$2, %eax	#,
	vdivsd	%xmm2, %xmm1, %xmm1	# tres2, tres1,
	vmovapd	%xmm2, %xmm0	# tres2,
	vmovsd	%xmm2, 8(%rsp)	# tres2, %sfp
.LVL61:
	call	__printf_chk@PLT	#
.LVL62:
.LBE112:
.LBE115:
# mulmatrix.c:123:     clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t0);
	.loc 2 123 0
	movq	%rbp, %rsi	# tmp161,
	movl	$3, %edi	#,
	call	clock_gettime@PLT	#
.LVL63:
# mulmatrix.c:125:     mulvec();
	.loc 2 125 0
	call	mulvec	#
.LVL64:
# mulmatrix.c:127:     clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t1);
	.loc 2 127 0
	movq	%rbx, %rsi	# tmp162,
	movl	$3, %edi	#,
	call	clock_gettime@PLT	#
.LVL65:
# mulmatrix.c:129:     tres3 = ((double)(t1.tv_sec - t0.tv_sec)*10e9 + (double)(t1.tv_nsec - t0.tv_nsec))/10e6;
	.loc 2 129 0
	movq	32(%rsp), %rax	# t1.tv_sec, t1.tv_sec
	subq	16(%rsp), %rax	# t0.tv_sec, tmp186
.LBB116:
.LBB117:
# /usr/include/x86_64-linux-gnu/bits/stdio2.h:104:   return __printf_chk (__USE_FORTIFY_LEVEL - 1, __fmt, __va_arg_pack ());
	.loc 6 104 0
	leaq	.LC6(%rip), %rsi	#,
.LBE117:
.LBE116:
# mulmatrix.c:129:     tres3 = ((double)(t1.tv_sec - t0.tv_sec)*10e9 + (double)(t1.tv_nsec - t0.tv_nsec))/10e6;
	.loc 2 129 0
	vxorpd	%xmm0, %xmm0, %xmm0	# tmp188
.LBB121:
.LBB118:
# /usr/include/x86_64-linux-gnu/bits/stdio2.h:104:   return __printf_chk (__USE_FORTIFY_LEVEL - 1, __fmt, __va_arg_pack ());
	.loc 6 104 0
	movl	$1, %edi	#,
.LBE118:
.LBE121:
# mulmatrix.c:129:     tres3 = ((double)(t1.tv_sec - t0.tv_sec)*10e9 + (double)(t1.tv_nsec - t0.tv_nsec))/10e6;
	.loc 2 129 0
	vxorpd	%xmm1, %xmm1, %xmm1	# tmp191
.LBB122:
.LBB119:
# /usr/include/x86_64-linux-gnu/bits/stdio2.h:104:   return __printf_chk (__USE_FORTIFY_LEVEL - 1, __fmt, __va_arg_pack ());
	.loc 6 104 0
	vmovsd	8(%rsp), %xmm2	# %sfp, tres2
.LBE119:
.LBE122:
# mulmatrix.c:129:     tres3 = ((double)(t1.tv_sec - t0.tv_sec)*10e9 + (double)(t1.tv_nsec - t0.tv_nsec))/10e6;
	.loc 2 129 0
	vcvtsi2sdq	%rax, %xmm0, %xmm0	# tmp186, tmp188, tmp188
	movq	40(%rsp), %rax	# t1.tv_nsec, t1.tv_nsec
	subq	24(%rsp), %rax	# t0.tv_nsec, tmp189
	vcvtsi2sdq	%rax, %xmm1, %xmm1	# tmp189, tmp191, tmp191
	vfmadd132sd	.LC2(%rip), %xmm1, %xmm0	#, tmp191, tmp192
	vdivsd	.LC3(%rip), %xmm0, %xmm0	#, tmp192, tres3
.LVL66:
.LBB123:
.LBB120:
# /usr/include/x86_64-linux-gnu/bits/stdio2.h:104:   return __printf_chk (__USE_FORTIFY_LEVEL - 1, __fmt, __va_arg_pack ());
	.loc 6 104 0
	movl	$2, %eax	#,
	vdivsd	%xmm0, %xmm2, %xmm1	# tres3, tres2,
	call	__printf_chk@PLT	#
.LVL67:
.LBE120:
.LBE123:
.LBB124:
# mulmatrix.c:133:             if(C2[i]!=C3[i]) {printf("Результирующие матрицы НЕ равны!\n"); break;}
	.loc 2 133 0
	movq	C2(%rip), %rcx	# C2, C2.13_50
	movq	C3(%rip), %rdx	# C3, C3.14_55
	xorl	%eax, %eax	# ivtmp.167
	jmp	.L58	#
.LVL68:
	.p2align 4,,10
	.p2align 3
.L64:
	addq	$8, %rax	#, ivtmp.167
# mulmatrix.c:132:     for (int i = 0; i < CR*CC; i++)
	.loc 2 132 0 discriminator 2
	cmpq	$8388608, %rax	#, ivtmp.167
	je	.L57	#,
.L58:
# mulmatrix.c:133:             if(C2[i]!=C3[i]) {printf("Результирующие матрицы НЕ равны!\n"); break;}
	.loc 2 133 0
	vmovsd	(%rcx,%rax), %xmm0	# MEM[base: C2.13_50, index: ivtmp.167_115, offset: 0B], MEM[base: C2.13_50, index: ivtmp.167_115, offset: 0B]
	vucomisd	(%rdx,%rax), %xmm0	# MEM[base: C3.14_55, index: ivtmp.167_115, offset: 0B], MEM[base: C2.13_50, index: ivtmp.167_115, offset: 0B]
	jp	.L60	#,
	je	.L64	#,
.L60:
.LVL69:
.LBB125:
.LBB126:
# /usr/include/x86_64-linux-gnu/bits/stdio2.h:104:   return __printf_chk (__USE_FORTIFY_LEVEL - 1, __fmt, __va_arg_pack ());
	.loc 6 104 0
	leaq	.LC7(%rip), %rdi	#,
	call	puts@PLT	#
.LVL70:
.L57:
.LBE126:
.LBE125:
.LBE124:
.LBB127:
.LBB128:
# /usr/lib/gcc/x86_64-linux-gnu/7/include/mm_malloc.h:54:   free (__ptr);
	.loc 1 54 0
	movq	A(%rip), %rdi	# A,
	call	free@PLT	#
.LVL71:
.LBE128:
.LBE127:
.LBB129:
.LBB130:
	movq	B(%rip), %rdi	# B,
	call	free@PLT	#
.LVL72:
.LBE130:
.LBE129:
.LBB131:
.LBB132:
	movq	C1(%rip), %rdi	# C1,
	call	free@PLT	#
.LVL73:
.LBE132:
.LBE131:
.LBB133:
.LBB134:
	movq	C2(%rip), %rdi	# C2,
	call	free@PLT	#
.LVL74:
.LBE134:
.LBE133:
.LBB135:
.LBB136:
	movq	C3(%rip), %rdi	# C3,
	call	free@PLT	#
.LVL75:
.LBE136:
.LBE135:
# mulmatrix.c:143: }
	.loc 2 143 0
	xorl	%eax, %eax	#
	movq	56(%rsp), %rsi	# D.29326, tmp206
	xorq	%fs:40, %rsi	#, tmp206
	jne	.L67	#,
	addq	$72, %rsp	#,
	.cfi_remember_state
	.cfi_def_cfa_offset 24
	popq	%rbx	#
	.cfi_def_cfa_offset 16
	popq	%rbp	#
	.cfi_def_cfa_offset 8
	ret
.L67:
	.cfi_restore_state
	call	__stack_chk_fail@PLT	#
.LVL76:
	.cfi_endproc
.LFE4795:
	.size	main, .-main
	.comm	C3,8,8
	.comm	C2,8,8
	.comm	C1,8,8
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
	.file 7 "/usr/lib/gcc/x86_64-linux-gnu/7/include/stddef.h"
	.file 8 "/usr/include/x86_64-linux-gnu/bits/types.h"
	.file 9 "/usr/include/x86_64-linux-gnu/bits/libio.h"
	.file 10 "/usr/include/stdio.h"
	.file 11 "/usr/include/x86_64-linux-gnu/bits/sys_errlist.h"
	.file 12 "/usr/include/x86_64-linux-gnu/bits/types/struct_timespec.h"
	.file 13 "/usr/include/time.h"
	.file 14 "/usr/include/malloc.h"
	.file 15 "/usr/include/stdlib.h"
	.file 16 "<built-in>"
	.section	.debug_info,"",@progbits
.Ldebug_info0:
	.long	0xe80
	.value	0x4
	.long	.Ldebug_abbrev0
	.byte	0x8
	.uleb128 0x1
	.long	.LASF106
	.byte	0xc
	.long	.LASF107
	.long	.LASF108
	.long	.Ldebug_ranges0+0x2f0
	.quad	0
	.long	.Ldebug_line0
	.uleb128 0x2
	.byte	0x8
	.byte	0x4
	.long	.LASF0
	.uleb128 0x3
	.long	.LASF8
	.byte	0x7
	.byte	0xd8
	.long	0x3b
	.uleb128 0x2
	.byte	0x8
	.byte	0x7
	.long	.LASF1
	.uleb128 0x2
	.byte	0x1
	.byte	0x8
	.long	.LASF2
	.uleb128 0x2
	.byte	0x2
	.byte	0x7
	.long	.LASF3
	.uleb128 0x2
	.byte	0x4
	.byte	0x7
	.long	.LASF4
	.uleb128 0x2
	.byte	0x1
	.byte	0x6
	.long	.LASF5
	.uleb128 0x2
	.byte	0x2
	.byte	0x5
	.long	.LASF6
	.uleb128 0x4
	.byte	0x4
	.byte	0x5
	.string	"int"
	.uleb128 0x2
	.byte	0x8
	.byte	0x5
	.long	.LASF7
	.uleb128 0x3
	.long	.LASF9
	.byte	0x8
	.byte	0x8c
	.long	0x6c
	.uleb128 0x3
	.long	.LASF10
	.byte	0x8
	.byte	0x8d
	.long	0x6c
	.uleb128 0x3
	.long	.LASF11
	.byte	0x8
	.byte	0x94
	.long	0x6c
	.uleb128 0x5
	.byte	0x8
	.uleb128 0x3
	.long	.LASF12
	.byte	0x8
	.byte	0xb8
	.long	0x6c
	.uleb128 0x6
	.byte	0x8
	.long	0xa7
	.uleb128 0x2
	.byte	0x1
	.byte	0x6
	.long	.LASF13
	.uleb128 0x7
	.long	0xa7
	.uleb128 0x8
	.long	.LASF43
	.byte	0xd8
	.byte	0x9
	.byte	0xf5
	.long	0x233
	.uleb128 0x9
	.long	.LASF14
	.byte	0x9
	.byte	0xf6
	.long	0x65
	.byte	0
	.uleb128 0x9
	.long	.LASF15
	.byte	0x9
	.byte	0xfb
	.long	0xa1
	.byte	0x8
	.uleb128 0x9
	.long	.LASF16
	.byte	0x9
	.byte	0xfc
	.long	0xa1
	.byte	0x10
	.uleb128 0x9
	.long	.LASF17
	.byte	0x9
	.byte	0xfd
	.long	0xa1
	.byte	0x18
	.uleb128 0x9
	.long	.LASF18
	.byte	0x9
	.byte	0xfe
	.long	0xa1
	.byte	0x20
	.uleb128 0x9
	.long	.LASF19
	.byte	0x9
	.byte	0xff
	.long	0xa1
	.byte	0x28
	.uleb128 0xa
	.long	.LASF20
	.byte	0x9
	.value	0x100
	.long	0xa1
	.byte	0x30
	.uleb128 0xa
	.long	.LASF21
	.byte	0x9
	.value	0x101
	.long	0xa1
	.byte	0x38
	.uleb128 0xa
	.long	.LASF22
	.byte	0x9
	.value	0x102
	.long	0xa1
	.byte	0x40
	.uleb128 0xa
	.long	.LASF23
	.byte	0x9
	.value	0x104
	.long	0xa1
	.byte	0x48
	.uleb128 0xa
	.long	.LASF24
	.byte	0x9
	.value	0x105
	.long	0xa1
	.byte	0x50
	.uleb128 0xa
	.long	.LASF25
	.byte	0x9
	.value	0x106
	.long	0xa1
	.byte	0x58
	.uleb128 0xa
	.long	.LASF26
	.byte	0x9
	.value	0x108
	.long	0x26b
	.byte	0x60
	.uleb128 0xa
	.long	.LASF27
	.byte	0x9
	.value	0x10a
	.long	0x271
	.byte	0x68
	.uleb128 0xa
	.long	.LASF28
	.byte	0x9
	.value	0x10c
	.long	0x65
	.byte	0x70
	.uleb128 0xa
	.long	.LASF29
	.byte	0x9
	.value	0x110
	.long	0x65
	.byte	0x74
	.uleb128 0xa
	.long	.LASF30
	.byte	0x9
	.value	0x112
	.long	0x73
	.byte	0x78
	.uleb128 0xa
	.long	.LASF31
	.byte	0x9
	.value	0x116
	.long	0x49
	.byte	0x80
	.uleb128 0xa
	.long	.LASF32
	.byte	0x9
	.value	0x117
	.long	0x57
	.byte	0x82
	.uleb128 0xa
	.long	.LASF33
	.byte	0x9
	.value	0x118
	.long	0x277
	.byte	0x83
	.uleb128 0xa
	.long	.LASF34
	.byte	0x9
	.value	0x11c
	.long	0x287
	.byte	0x88
	.uleb128 0xa
	.long	.LASF35
	.byte	0x9
	.value	0x125
	.long	0x7e
	.byte	0x90
	.uleb128 0xa
	.long	.LASF36
	.byte	0x9
	.value	0x12d
	.long	0x94
	.byte	0x98
	.uleb128 0xa
	.long	.LASF37
	.byte	0x9
	.value	0x12e
	.long	0x94
	.byte	0xa0
	.uleb128 0xa
	.long	.LASF38
	.byte	0x9
	.value	0x12f
	.long	0x94
	.byte	0xa8
	.uleb128 0xa
	.long	.LASF39
	.byte	0x9
	.value	0x130
	.long	0x94
	.byte	0xb0
	.uleb128 0xa
	.long	.LASF40
	.byte	0x9
	.value	0x132
	.long	0x30
	.byte	0xb8
	.uleb128 0xa
	.long	.LASF41
	.byte	0x9
	.value	0x133
	.long	0x65
	.byte	0xc0
	.uleb128 0xa
	.long	.LASF42
	.byte	0x9
	.value	0x135
	.long	0x28d
	.byte	0xc4
	.byte	0
	.uleb128 0xb
	.long	.LASF109
	.byte	0x9
	.byte	0x9a
	.uleb128 0x8
	.long	.LASF44
	.byte	0x18
	.byte	0x9
	.byte	0xa0
	.long	0x26b
	.uleb128 0x9
	.long	.LASF45
	.byte	0x9
	.byte	0xa1
	.long	0x26b
	.byte	0
	.uleb128 0x9
	.long	.LASF46
	.byte	0x9
	.byte	0xa2
	.long	0x271
	.byte	0x8
	.uleb128 0x9
	.long	.LASF47
	.byte	0x9
	.byte	0xa6
	.long	0x65
	.byte	0x10
	.byte	0
	.uleb128 0x6
	.byte	0x8
	.long	0x23a
	.uleb128 0x6
	.byte	0x8
	.long	0xb3
	.uleb128 0xc
	.long	0xa7
	.long	0x287
	.uleb128 0xd
	.long	0x3b
	.byte	0
	.byte	0
	.uleb128 0x6
	.byte	0x8
	.long	0x233
	.uleb128 0xc
	.long	0xa7
	.long	0x29d
	.uleb128 0xd
	.long	0x3b
	.byte	0x13
	.byte	0
	.uleb128 0xe
	.long	.LASF110
	.uleb128 0xf
	.long	.LASF48
	.byte	0x9
	.value	0x13f
	.long	0x29d
	.uleb128 0xf
	.long	.LASF49
	.byte	0x9
	.value	0x140
	.long	0x29d
	.uleb128 0xf
	.long	.LASF50
	.byte	0x9
	.value	0x141
	.long	0x29d
	.uleb128 0x6
	.byte	0x8
	.long	0xae
	.uleb128 0x7
	.long	0x2c6
	.uleb128 0x10
	.long	0x2c6
	.uleb128 0x11
	.long	.LASF51
	.byte	0xa
	.byte	0x87
	.long	0x271
	.uleb128 0x11
	.long	.LASF52
	.byte	0xa
	.byte	0x88
	.long	0x271
	.uleb128 0x11
	.long	.LASF53
	.byte	0xa
	.byte	0x89
	.long	0x271
	.uleb128 0x11
	.long	.LASF54
	.byte	0xb
	.byte	0x1a
	.long	0x65
	.uleb128 0xc
	.long	0x2cc
	.long	0x30d
	.uleb128 0x12
	.byte	0
	.uleb128 0x7
	.long	0x302
	.uleb128 0x11
	.long	.LASF55
	.byte	0xb
	.byte	0x1b
	.long	0x30d
	.uleb128 0x8
	.long	.LASF56
	.byte	0x10
	.byte	0xc
	.byte	0x8
	.long	0x342
	.uleb128 0x9
	.long	.LASF57
	.byte	0xc
	.byte	0xa
	.long	0x89
	.byte	0
	.uleb128 0x9
	.long	.LASF58
	.byte	0xc
	.byte	0xb
	.long	0x96
	.byte	0x8
	.byte	0
	.uleb128 0xc
	.long	0xa1
	.long	0x352
	.uleb128 0xd
	.long	0x3b
	.byte	0x1
	.byte	0
	.uleb128 0x11
	.long	.LASF59
	.byte	0xd
	.byte	0x9f
	.long	0x342
	.uleb128 0x11
	.long	.LASF60
	.byte	0xd
	.byte	0xa0
	.long	0x65
	.uleb128 0x11
	.long	.LASF61
	.byte	0xd
	.byte	0xa1
	.long	0x6c
	.uleb128 0x11
	.long	.LASF62
	.byte	0xd
	.byte	0xa6
	.long	0x342
	.uleb128 0x11
	.long	.LASF63
	.byte	0xd
	.byte	0xae
	.long	0x65
	.uleb128 0x11
	.long	.LASF64
	.byte	0xd
	.byte	0xaf
	.long	0x6c
	.uleb128 0x3
	.long	.LASF65
	.byte	0x7
	.byte	0x95
	.long	0x6c
	.uleb128 0x2
	.byte	0x8
	.byte	0x5
	.long	.LASF66
	.uleb128 0x2
	.byte	0x10
	.byte	0x4
	.long	.LASF67
	.uleb128 0x13
	.long	0x94
	.long	0x3bc
	.uleb128 0x14
	.long	0x394
	.byte	0
	.uleb128 0x11
	.long	.LASF68
	.byte	0xe
	.byte	0x4c
	.long	0x3c7
	.uleb128 0x6
	.byte	0x8
	.long	0x3ad
	.uleb128 0x15
	.long	0x3dd
	.uleb128 0x14
	.long	0x94
	.uleb128 0x14
	.long	0x3dd
	.byte	0
	.uleb128 0x6
	.byte	0x8
	.long	0x3e3
	.uleb128 0x16
	.uleb128 0x11
	.long	.LASF69
	.byte	0xe
	.byte	0x8f
	.long	0x3f5
	.uleb128 0x6
	.byte	0x8
	.long	0x3cd
	.uleb128 0x17
	.long	0x3ef
	.uleb128 0x13
	.long	0x94
	.long	0x40e
	.uleb128 0x14
	.long	0x30
	.uleb128 0x14
	.long	0x3dd
	.byte	0
	.uleb128 0x11
	.long	.LASF70
	.byte	0xe
	.byte	0x92
	.long	0x41f
	.uleb128 0x6
	.byte	0x8
	.long	0x3fa
	.uleb128 0x17
	.long	0x419
	.uleb128 0x13
	.long	0x94
	.long	0x43d
	.uleb128 0x14
	.long	0x94
	.uleb128 0x14
	.long	0x30
	.uleb128 0x14
	.long	0x3dd
	.byte	0
	.uleb128 0x11
	.long	.LASF71
	.byte	0xe
	.byte	0x95
	.long	0x44e
	.uleb128 0x6
	.byte	0x8
	.long	0x424
	.uleb128 0x17
	.long	0x448
	.uleb128 0x13
	.long	0x94
	.long	0x46c
	.uleb128 0x14
	.long	0x30
	.uleb128 0x14
	.long	0x30
	.uleb128 0x14
	.long	0x3dd
	.byte	0
	.uleb128 0x11
	.long	.LASF72
	.byte	0xe
	.byte	0x99
	.long	0x47d
	.uleb128 0x6
	.byte	0x8
	.long	0x453
	.uleb128 0x17
	.long	0x477
	.uleb128 0x18
	.uleb128 0x11
	.long	.LASF73
	.byte	0xe
	.byte	0x9d
	.long	0x494
	.uleb128 0x6
	.byte	0x8
	.long	0x482
	.uleb128 0x17
	.long	0x48e
	.uleb128 0x2
	.byte	0x8
	.byte	0x7
	.long	.LASF74
	.uleb128 0x2
	.byte	0x4
	.byte	0x4
	.long	.LASF75
	.uleb128 0x7
	.long	0x29
	.uleb128 0x3
	.long	.LASF76
	.byte	0x4
	.byte	0x29
	.long	0x4b7
	.uleb128 0x19
	.long	0x29
	.long	0x4c3
	.uleb128 0x1a
	.byte	0x3
	.byte	0
	.uleb128 0x3
	.long	.LASF77
	.byte	0x4
	.byte	0x3a
	.long	0x4ce
	.uleb128 0x19
	.long	0x29
	.long	0x4da
	.uleb128 0x1a
	.byte	0x3
	.byte	0
	.uleb128 0x1b
	.string	"A"
	.byte	0x2
	.byte	0x10
	.long	0x4ed
	.uleb128 0x9
	.byte	0x3
	.quad	A
	.uleb128 0x6
	.byte	0x8
	.long	0x29
	.uleb128 0x1b
	.string	"B"
	.byte	0x2
	.byte	0x11
	.long	0x4ed
	.uleb128 0x9
	.byte	0x3
	.quad	B
	.uleb128 0x1b
	.string	"C1"
	.byte	0x2
	.byte	0x12
	.long	0x4ed
	.uleb128 0x9
	.byte	0x3
	.quad	C1
	.uleb128 0x1b
	.string	"C2"
	.byte	0x2
	.byte	0x13
	.long	0x4ed
	.uleb128 0x9
	.byte	0x3
	.quad	C2
	.uleb128 0x1b
	.string	"C3"
	.byte	0x2
	.byte	0x14
	.long	0x4ed
	.uleb128 0x9
	.byte	0x3
	.quad	C3
	.uleb128 0x1c
	.long	.LASF111
	.byte	0x2
	.byte	0x54
	.long	0x65
	.quad	.LFB4795
	.quad	.LFE4795-.LFB4795
	.uleb128 0x1
	.byte	0x9c
	.long	0x968
	.uleb128 0x1d
	.long	.LASF78
	.byte	0x2
	.byte	0x54
	.long	0x65
	.long	.LLST19
	.uleb128 0x1d
	.long	.LASF79
	.byte	0x2
	.byte	0x54
	.long	0x968
	.long	.LLST20
	.uleb128 0x1e
	.string	"t0"
	.byte	0x2
	.byte	0x66
	.long	0x31d
	.uleb128 0x3
	.byte	0x91
	.sleb128 -80
	.uleb128 0x1e
	.string	"t1"
	.byte	0x2
	.byte	0x66
	.long	0x31d
	.uleb128 0x2
	.byte	0x91
	.sleb128 -64
	.uleb128 0x1f
	.long	.LASF80
	.byte	0x2
	.byte	0x67
	.long	0x29
	.long	.LLST21
	.uleb128 0x1f
	.long	.LASF81
	.byte	0x2
	.byte	0x67
	.long	0x29
	.long	.LLST22
	.uleb128 0x1f
	.long	.LASF82
	.byte	0x2
	.byte	0x67
	.long	0x29
	.long	.LLST23
	.uleb128 0x20
	.quad	.LBB100
	.quad	.LBE100-.LBB100
	.long	0x5f5
	.uleb128 0x21
	.string	"i"
	.byte	0x2
	.byte	0x5c
	.long	0x65
	.uleb128 0x22
	.quad	.LVL50
	.long	0xde1
	.byte	0
	.uleb128 0x20
	.quad	.LBB101
	.quad	.LBE101-.LBB101
	.long	0x621
	.uleb128 0x21
	.string	"i"
	.byte	0x2
	.byte	0x61
	.long	0x65
	.uleb128 0x22
	.quad	.LVL51
	.long	0xde1
	.byte	0
	.uleb128 0x20
	.quad	.LBB124
	.quad	.LBE124-.LBB124
	.long	0x680
	.uleb128 0x23
	.string	"i"
	.byte	0x2
	.byte	0x84
	.long	0x65
	.long	.LLST27
	.uleb128 0x24
	.long	0xd62
	.quad	.LBB125
	.quad	.LBE125-.LBB125
	.byte	0x2
	.byte	0x85
	.uleb128 0x25
	.long	0xd72
	.long	.LLST28
	.uleb128 0x26
	.quad	.LVL70
	.long	0xe2f
	.uleb128 0x27
	.uleb128 0x1
	.byte	0x55
	.uleb128 0x9
	.byte	0x3
	.quad	.LC7
	.byte	0
	.byte	0
	.byte	0
	.uleb128 0x28
	.long	0xd62
	.quad	.LBB102
	.long	.Ldebug_ranges0+0x210
	.byte	0x2
	.byte	0x70
	.long	0x6cb
	.uleb128 0x25
	.long	0xd72
	.long	.LLST24
	.uleb128 0x26
	.quad	.LVL56
	.long	0xe3e
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
	.uleb128 0x27
	.uleb128 0x1
	.byte	0x61
	.uleb128 0x6
	.byte	0x91
	.sleb128 -88
	.byte	0xf6
	.byte	0x8
	.uleb128 0x29
	.byte	0
	.byte	0
	.uleb128 0x28
	.long	0xd62
	.quad	.LBB108
	.long	.Ldebug_ranges0+0x250
	.byte	0x2
	.byte	0x79
	.long	0x716
	.uleb128 0x25
	.long	0xd72
	.long	.LLST25
	.uleb128 0x26
	.quad	.LVL62
	.long	0xe3e
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
	.uleb128 0x27
	.uleb128 0x1
	.byte	0x61
	.uleb128 0x6
	.byte	0x91
	.sleb128 -88
	.byte	0xf6
	.byte	0x8
	.uleb128 0x29
	.byte	0
	.byte	0
	.uleb128 0x28
	.long	0xd62
	.quad	.LBB116
	.long	.Ldebug_ranges0+0x2a0
	.byte	0x2
	.byte	0x82
	.long	0x757
	.uleb128 0x25
	.long	0xd72
	.long	.LLST26
	.uleb128 0x26
	.quad	.LVL67
	.long	0xe3e
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
	.uleb128 0x29
	.long	0xce6
	.quad	.LBB127
	.quad	.LBE127-.LBB127
	.byte	0x2
	.byte	0x87
	.long	0x789
	.uleb128 0x25
	.long	0xcf2
	.long	.LLST29
	.uleb128 0x22
	.quad	.LVL71
	.long	0xe49
	.byte	0
	.uleb128 0x29
	.long	0xce6
	.quad	.LBB129
	.quad	.LBE129-.LBB129
	.byte	0x2
	.byte	0x88
	.long	0x7bb
	.uleb128 0x25
	.long	0xcf2
	.long	.LLST30
	.uleb128 0x22
	.quad	.LVL72
	.long	0xe49
	.byte	0
	.uleb128 0x29
	.long	0xce6
	.quad	.LBB131
	.quad	.LBE131-.LBB131
	.byte	0x2
	.byte	0x89
	.long	0x7ed
	.uleb128 0x25
	.long	0xcf2
	.long	.LLST31
	.uleb128 0x22
	.quad	.LVL73
	.long	0xe49
	.byte	0
	.uleb128 0x29
	.long	0xce6
	.quad	.LBB133
	.quad	.LBE133-.LBB133
	.byte	0x2
	.byte	0x8a
	.long	0x81f
	.uleb128 0x25
	.long	0xcf2
	.long	.LLST32
	.uleb128 0x22
	.quad	.LVL74
	.long	0xe49
	.byte	0
	.uleb128 0x29
	.long	0xce6
	.quad	.LBB135
	.quad	.LBE135-.LBB135
	.byte	0x2
	.byte	0x8b
	.long	0x851
	.uleb128 0x25
	.long	0xcf2
	.long	.LLST33
	.uleb128 0x22
	.quad	.LVL75
	.long	0xe49
	.byte	0
	.uleb128 0x22
	.quad	.LVL44
	.long	0xd7f
	.uleb128 0x22
	.quad	.LVL45
	.long	0xd7f
	.uleb128 0x22
	.quad	.LVL46
	.long	0xd7f
	.uleb128 0x22
	.quad	.LVL47
	.long	0xd7f
	.uleb128 0x22
	.quad	.LVL48
	.long	0xd7f
	.uleb128 0x2a
	.quad	.LVL53
	.long	0xe55
	.long	0x8af
	.uleb128 0x27
	.uleb128 0x1
	.byte	0x55
	.uleb128 0x1
	.byte	0x33
	.uleb128 0x27
	.uleb128 0x1
	.byte	0x54
	.uleb128 0x2
	.byte	0x76
	.sleb128 0
	.byte	0
	.uleb128 0x2a
	.quad	.LVL54
	.long	0xe55
	.long	0x8cc
	.uleb128 0x27
	.uleb128 0x1
	.byte	0x55
	.uleb128 0x1
	.byte	0x33
	.uleb128 0x27
	.uleb128 0x1
	.byte	0x54
	.uleb128 0x2
	.byte	0x73
	.sleb128 0
	.byte	0
	.uleb128 0x2a
	.quad	.LVL57
	.long	0xe55
	.long	0x8e9
	.uleb128 0x27
	.uleb128 0x1
	.byte	0x55
	.uleb128 0x1
	.byte	0x33
	.uleb128 0x27
	.uleb128 0x1
	.byte	0x54
	.uleb128 0x2
	.byte	0x76
	.sleb128 0
	.byte	0
	.uleb128 0x22
	.quad	.LVL58
	.long	0xb66
	.uleb128 0x2a
	.quad	.LVL59
	.long	0xe55
	.long	0x913
	.uleb128 0x27
	.uleb128 0x1
	.byte	0x55
	.uleb128 0x1
	.byte	0x33
	.uleb128 0x27
	.uleb128 0x1
	.byte	0x54
	.uleb128 0x2
	.byte	0x73
	.sleb128 0
	.byte	0
	.uleb128 0x2a
	.quad	.LVL63
	.long	0xe55
	.long	0x930
	.uleb128 0x27
	.uleb128 0x1
	.byte	0x55
	.uleb128 0x1
	.byte	0x33
	.uleb128 0x27
	.uleb128 0x1
	.byte	0x54
	.uleb128 0x2
	.byte	0x76
	.sleb128 0
	.byte	0
	.uleb128 0x22
	.quad	.LVL64
	.long	0x96e
	.uleb128 0x2a
	.quad	.LVL65
	.long	0xe55
	.long	0x95a
	.uleb128 0x27
	.uleb128 0x1
	.byte	0x55
	.uleb128 0x1
	.byte	0x33
	.uleb128 0x27
	.uleb128 0x1
	.byte	0x54
	.uleb128 0x2
	.byte	0x73
	.sleb128 0
	.byte	0
	.uleb128 0x22
	.quad	.LVL76
	.long	0xe60
	.byte	0
	.uleb128 0x6
	.byte	0x8
	.long	0x2c6
	.uleb128 0x2b
	.long	.LASF83
	.byte	0x2
	.byte	0x37
	.quad	.LFB4794
	.quad	.LFE4794-.LFB4794
	.uleb128 0x1
	.byte	0x9c
	.long	0xb60
	.uleb128 0x20
	.quad	.LBB85
	.quad	.LBE85-.LBB85
	.long	0xb16
	.uleb128 0x23
	.string	"i"
	.byte	0x2
	.byte	0x39
	.long	0x65
	.long	.LLST11
	.uleb128 0x2c
	.quad	.LBB86
	.quad	.LBE86-.LBB86
	.uleb128 0x1e
	.string	"c"
	.byte	0x2
	.byte	0x3a
	.long	0x4ed
	.uleb128 0x1
	.byte	0x58
	.uleb128 0x1e
	.string	"ap"
	.byte	0x2
	.byte	0x3b
	.long	0x4ed
	.uleb128 0x1
	.byte	0x54
	.uleb128 0x23
	.string	"b"
	.byte	0x2
	.byte	0x3c
	.long	0xb60
	.long	.LLST12
	.uleb128 0x2c
	.quad	.LBB87
	.quad	.LBE87-.LBB87
	.uleb128 0x23
	.string	"k"
	.byte	0x2
	.byte	0x3e
	.long	0x65
	.long	.LLST13
	.uleb128 0x2c
	.quad	.LBB88
	.quad	.LBE88-.LBB88
	.uleb128 0x21
	.string	"a"
	.byte	0x2
	.byte	0x40
	.long	0x4c3
	.uleb128 0x23
	.string	"cp"
	.byte	0x2
	.byte	0x41
	.long	0x4ed
	.long	.LLST14
	.uleb128 0x20
	.quad	.LBB91
	.quad	.LBE91-.LBB91
	.long	0xaf1
	.uleb128 0x23
	.string	"j"
	.byte	0x2
	.byte	0x42
	.long	0x65
	.long	.LLST16
	.uleb128 0x29
	.long	0xc52
	.quad	.LBB92
	.quad	.LBE92-.LBB92
	.byte	0x2
	.byte	0x43
	.long	0xa79
	.uleb128 0x2d
	.long	0xc78
	.uleb128 0x2d
	.long	0xc6d
	.uleb128 0x25
	.long	0xc62
	.long	.LLST17
	.byte	0
	.uleb128 0x29
	.long	0xc84
	.quad	.LBB94
	.quad	.LBE94-.LBB94
	.byte	0x2
	.byte	0x43
	.long	0xa9f
	.uleb128 0x2d
	.long	0xc9d
	.uleb128 0x2d
	.long	0xc91
	.byte	0
	.uleb128 0x29
	.long	0xc52
	.quad	.LBB96
	.quad	.LBE96-.LBB96
	.byte	0x2
	.byte	0x48
	.long	0xace
	.uleb128 0x2d
	.long	0xc78
	.uleb128 0x2d
	.long	0xc6d
	.uleb128 0x25
	.long	0xc62
	.long	.LLST18
	.byte	0
	.uleb128 0x24
	.long	0xc84
	.quad	.LBB98
	.quad	.LBE98-.LBB98
	.byte	0x2
	.byte	0x48
	.uleb128 0x2d
	.long	0xc9d
	.uleb128 0x2d
	.long	0xc91
	.byte	0
	.byte	0
	.uleb128 0x24
	.long	0xcc8
	.quad	.LBB89
	.quad	.LBE89-.LBB89
	.byte	0x2
	.byte	0x40
	.uleb128 0x25
	.long	0xcd9
	.long	.LLST15
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.uleb128 0x2e
	.long	0xd30
	.quad	.LBB81
	.long	.Ldebug_ranges0+0x1e0
	.byte	0x2
	.byte	0x38
	.uleb128 0x25
	.long	0xd56
	.long	.LLST8
	.uleb128 0x25
	.long	0xd4b
	.long	.LLST9
	.uleb128 0x25
	.long	0xd40
	.long	.LLST10
	.uleb128 0x26
	.quad	.LVL29
	.long	0xe69
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
	.uleb128 0x2b
	.long	.LASF84
	.byte	0x2
	.byte	0x24
	.quad	.LFB4793
	.quad	.LFE4793-.LFB4793
	.uleb128 0x1
	.byte	0x9c
	.long	0xbfb
	.uleb128 0x2f
	.long	.Ldebug_ranges0+0x90
	.uleb128 0x23
	.string	"i"
	.byte	0x2
	.byte	0x25
	.long	0x65
	.long	.LLST3
	.uleb128 0x2f
	.long	.Ldebug_ranges0+0xc0
	.uleb128 0x23
	.string	"c"
	.byte	0x2
	.byte	0x26
	.long	0x4ed
	.long	.LLST4
	.uleb128 0x30
	.long	0xbb6
	.uleb128 0x21
	.string	"j"
	.byte	0x2
	.byte	0x28
	.long	0x65
	.byte	0
	.uleb128 0x2f
	.long	.Ldebug_ranges0+0xf0
	.uleb128 0x21
	.string	"k"
	.byte	0x2
	.byte	0x2b
	.long	0x65
	.uleb128 0x2f
	.long	.Ldebug_ranges0+0x140
	.uleb128 0x23
	.string	"b"
	.byte	0x2
	.byte	0x2c
	.long	0xb60
	.long	.LLST5
	.uleb128 0x23
	.string	"a"
	.byte	0x2
	.byte	0x2d
	.long	0x29
	.long	.LLST6
	.uleb128 0x2f
	.long	.Ldebug_ranges0+0x190
	.uleb128 0x23
	.string	"j"
	.byte	0x2
	.byte	0x2f
	.long	0x65
	.long	.LLST7
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.uleb128 0x2b
	.long	.LASF85
	.byte	0x2
	.byte	0x16
	.quad	.LFB4792
	.quad	.LFE4792-.LFB4792
	.uleb128 0x1
	.byte	0x9c
	.long	0xc52
	.uleb128 0x2f
	.long	.Ldebug_ranges0+0
	.uleb128 0x23
	.string	"i"
	.byte	0x2
	.byte	0x17
	.long	0x65
	.long	.LLST0
	.uleb128 0x2f
	.long	.Ldebug_ranges0+0x30
	.uleb128 0x23
	.string	"j"
	.byte	0x2
	.byte	0x19
	.long	0x65
	.long	.LLST1
	.uleb128 0x2f
	.long	.Ldebug_ranges0+0x60
	.uleb128 0x23
	.string	"k"
	.byte	0x2
	.byte	0x1c
	.long	0x65
	.long	.LLST2
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.uleb128 0x31
	.long	.LASF86
	.byte	0x5
	.byte	0x2f
	.long	0x4c3
	.byte	0x3
	.long	0xc84
	.uleb128 0x32
	.string	"__A"
	.byte	0x5
	.byte	0x2f
	.long	0x4c3
	.uleb128 0x32
	.string	"__B"
	.byte	0x5
	.byte	0x2f
	.long	0x4c3
	.uleb128 0x32
	.string	"__C"
	.byte	0x5
	.byte	0x2f
	.long	0x4c3
	.byte	0
	.uleb128 0x33
	.long	.LASF112
	.byte	0x4
	.value	0x361
	.byte	0x3
	.long	0xcaa
	.uleb128 0x34
	.string	"__P"
	.byte	0x4
	.value	0x361
	.long	0x4ed
	.uleb128 0x34
	.string	"__A"
	.byte	0x4
	.value	0x361
	.long	0x4c3
	.byte	0
	.uleb128 0x35
	.long	.LASF87
	.byte	0x4
	.value	0x35b
	.long	0x4c3
	.byte	0x3
	.long	0xcc8
	.uleb128 0x34
	.string	"__P"
	.byte	0x4
	.value	0x35b
	.long	0xb60
	.byte	0
	.uleb128 0x35
	.long	.LASF88
	.byte	0x4
	.value	0x2d6
	.long	0x4c3
	.byte	0x3
	.long	0xce6
	.uleb128 0x34
	.string	"__X"
	.byte	0x4
	.value	0x2d6
	.long	0xb60
	.byte	0
	.uleb128 0x36
	.long	.LASF113
	.byte	0x1
	.byte	0x34
	.byte	0x3
	.long	0xcfe
	.uleb128 0x37
	.long	.LASF89
	.byte	0x1
	.byte	0x34
	.long	0x94
	.byte	0
	.uleb128 0x38
	.long	.LASF114
	.byte	0x1
	.byte	0x26
	.long	0x94
	.byte	0x3
	.long	0xd30
	.uleb128 0x37
	.long	.LASF90
	.byte	0x1
	.byte	0x26
	.long	0x30
	.uleb128 0x37
	.long	.LASF91
	.byte	0x1
	.byte	0x26
	.long	0x30
	.uleb128 0x39
	.long	.LASF89
	.byte	0x1
	.byte	0x28
	.long	0x94
	.byte	0
	.uleb128 0x31
	.long	.LASF92
	.byte	0x3
	.byte	0x3b
	.long	0x94
	.byte	0x3
	.long	0xd62
	.uleb128 0x37
	.long	.LASF93
	.byte	0x3
	.byte	0x3b
	.long	0x94
	.uleb128 0x37
	.long	.LASF94
	.byte	0x3
	.byte	0x3b
	.long	0x65
	.uleb128 0x37
	.long	.LASF95
	.byte	0x3
	.byte	0x3b
	.long	0x30
	.byte	0
	.uleb128 0x31
	.long	.LASF96
	.byte	0x6
	.byte	0x66
	.long	0x65
	.byte	0x3
	.long	0xd7f
	.uleb128 0x37
	.long	.LASF97
	.byte	0x6
	.byte	0x66
	.long	0x2d1
	.uleb128 0x3a
	.byte	0
	.uleb128 0x3b
	.long	0xcfe
	.quad	.LFB4796
	.quad	.LFE4796-.LFB4796
	.uleb128 0x1
	.byte	0x9c
	.long	0xde1
	.uleb128 0x3c
	.long	0xd24
	.uleb128 0x3d
	.long	0xd19
	.byte	0x20
	.uleb128 0x3e
	.long	0xd0e
	.long	0x800000
	.uleb128 0x2a
	.quad	.LVL1
	.long	0xe78
	.long	0xdd3
	.uleb128 0x27
	.uleb128 0x1
	.byte	0x55
	.uleb128 0x2
	.byte	0x77
	.sleb128 0
	.uleb128 0x27
	.uleb128 0x1
	.byte	0x54
	.uleb128 0x2
	.byte	0x8
	.byte	0x20
	.uleb128 0x27
	.uleb128 0x1
	.byte	0x51
	.uleb128 0x3
	.byte	0x40
	.byte	0x43
	.byte	0x24
	.byte	0
	.uleb128 0x22
	.quad	.LVL2
	.long	0xe60
	.byte	0
	.uleb128 0x3f
	.long	.LASF98
	.long	.LASF98
	.byte	0xf
	.value	0x1d3
	.uleb128 0x40
	.uleb128 0x40
	.byte	0x9e
	.uleb128 0x3e
	.byte	0xd0
	.byte	0xa0
	.byte	0xd0
	.byte	0xb5
	.byte	0xd0
	.byte	0xb7
	.byte	0xd1
	.byte	0x83
	.byte	0xd0
	.byte	0xbb
	.byte	0xd1
	.byte	0x8c
	.byte	0xd1
	.byte	0x82
	.byte	0xd0
	.byte	0xb8
	.byte	0xd1
	.byte	0x80
	.byte	0xd1
	.byte	0x83
	.byte	0xd1
	.byte	0x8e
	.byte	0xd1
	.byte	0x89
	.byte	0xd0
	.byte	0xb8
	.byte	0xd0
	.byte	0xb5
	.byte	0x20
	.byte	0xd0
	.byte	0xbc
	.byte	0xd0
	.byte	0xb0
	.byte	0xd1
	.byte	0x82
	.byte	0xd1
	.byte	0x80
	.byte	0xd0
	.byte	0xb8
	.byte	0xd1
	.byte	0x86
	.byte	0xd1
	.byte	0x8b
	.byte	0x20
	.byte	0xd0
	.byte	0x9d
	.byte	0xd0
	.byte	0x95
	.byte	0x20
	.byte	0xd1
	.byte	0x80
	.byte	0xd0
	.byte	0xb0
	.byte	0xd0
	.byte	0xb2
	.byte	0xd0
	.byte	0xbd
	.byte	0xd1
	.byte	0x8b
	.byte	0x21
	.byte	0xa
	.byte	0
	.uleb128 0x41
	.long	.LASF102
	.long	.LASF103
	.byte	0x10
	.byte	0
	.long	.LASF102
	.uleb128 0x42
	.long	.LASF99
	.long	.LASF99
	.byte	0x6
	.byte	0x57
	.uleb128 0x3f
	.long	.LASF100
	.long	.LASF100
	.byte	0xf
	.value	0x233
	.uleb128 0x42
	.long	.LASF101
	.long	.LASF101
	.byte	0xd
	.byte	0xdb
	.uleb128 0x43
	.long	.LASF115
	.long	.LASF115
	.uleb128 0x41
	.long	.LASF92
	.long	.LASF104
	.byte	0x10
	.byte	0
	.long	.LASF92
	.uleb128 0x42
	.long	.LASF105
	.long	.LASF105
	.byte	0x1
	.byte	0x20
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
	.uleb128 0x3
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
	.uleb128 0x1c
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
	.uleb128 0x1d
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
	.uleb128 0x1e
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
	.uleb128 0x1f
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
	.uleb128 0x2
	.uleb128 0x17
	.byte	0
	.byte	0
	.uleb128 0x20
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
	.uleb128 0x21
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
	.uleb128 0x22
	.uleb128 0x4109
	.byte	0
	.uleb128 0x11
	.uleb128 0x1
	.uleb128 0x31
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x23
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
	.uleb128 0x24
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
	.uleb128 0x29
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
	.uleb128 0x2a
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
	.uleb128 0x2b
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
	.uleb128 0x2c
	.uleb128 0xb
	.byte	0x1
	.uleb128 0x11
	.uleb128 0x1
	.uleb128 0x12
	.uleb128 0x7
	.byte	0
	.byte	0
	.uleb128 0x2d
	.uleb128 0x5
	.byte	0
	.uleb128 0x31
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x2e
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
	.uleb128 0x2f
	.uleb128 0xb
	.byte	0x1
	.uleb128 0x55
	.uleb128 0x17
	.byte	0
	.byte	0
	.uleb128 0x30
	.uleb128 0xb
	.byte	0x1
	.uleb128 0x1
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x31
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
	.uleb128 0x32
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
	.uleb128 0x33
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
	.uleb128 0x34
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
	.uleb128 0x35
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
	.uleb128 0x36
	.uleb128 0x2e
	.byte	0x1
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x27
	.uleb128 0x19
	.uleb128 0x20
	.uleb128 0xb
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
	.uleb128 0x2e
	.byte	0x1
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
	.uleb128 0x1
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x39
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
	.uleb128 0x3a
	.uleb128 0x18
	.byte	0
	.byte	0
	.byte	0
	.uleb128 0x3b
	.uleb128 0x2e
	.byte	0x1
	.uleb128 0x31
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
	.uleb128 0x3c
	.uleb128 0x34
	.byte	0
	.uleb128 0x31
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x3d
	.uleb128 0x5
	.byte	0
	.uleb128 0x31
	.uleb128 0x13
	.uleb128 0x1c
	.uleb128 0xb
	.byte	0
	.byte	0
	.uleb128 0x3e
	.uleb128 0x5
	.byte	0
	.uleb128 0x31
	.uleb128 0x13
	.uleb128 0x1c
	.uleb128 0x6
	.byte	0
	.byte	0
	.uleb128 0x3f
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
	.uleb128 0x40
	.uleb128 0x36
	.byte	0
	.uleb128 0x2
	.uleb128 0x18
	.byte	0
	.byte	0
	.uleb128 0x41
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
	.uleb128 0x42
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
	.uleb128 0x43
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
	.byte	0
	.section	.debug_loc,"",@progbits
.Ldebug_loc0:
.LLST19:
	.quad	.LVL43
	.quad	.LVL44-1
	.value	0x1
	.byte	0x55
	.quad	.LVL44-1
	.quad	.LFE4795
	.value	0x4
	.byte	0xf3
	.uleb128 0x1
	.byte	0x55
	.byte	0x9f
	.quad	0
	.quad	0
.LLST20:
	.quad	.LVL43
	.quad	.LVL44-1
	.value	0x1
	.byte	0x54
	.quad	.LVL44-1
	.quad	.LFE4795
	.value	0x4
	.byte	0xf3
	.uleb128 0x1
	.byte	0x54
	.byte	0x9f
	.quad	0
	.quad	0
.LLST21:
	.quad	.LVL55
	.quad	.LVL56-1
	.value	0x1
	.byte	0x62
	.quad	.LVL56-1
	.quad	.LVL61
	.value	0x3
	.byte	0x91
	.sleb128 -88
	.quad	0
	.quad	0
.LLST22:
	.quad	.LVL60
	.quad	.LVL62-1
	.value	0x1
	.byte	0x63
	.quad	.LVL62-1
	.quad	.LFE4795
	.value	0x3
	.byte	0x91
	.sleb128 -88
	.quad	0
	.quad	0
.LLST23:
	.quad	.LVL66
	.quad	.LVL67-1
	.value	0x1
	.byte	0x61
	.quad	0
	.quad	0
.LLST27:
	.quad	.LVL67
	.quad	.LVL68
	.value	0x2
	.byte	0x30
	.byte	0x9f
	.quad	0
	.quad	0
.LLST28:
	.quad	.LVL69
	.quad	.LVL70
	.value	0x6
	.byte	0xf2
	.long	.Ldebug_info0+3565
	.sleb128 0
	.quad	0
	.quad	0
.LLST24:
	.quad	.LVL55
	.quad	.LVL56
	.value	0xa
	.byte	0x3
	.quad	.LC4
	.byte	0x9f
	.quad	0
	.quad	0
.LLST25:
	.quad	.LVL60
	.quad	.LVL62
	.value	0xa
	.byte	0x3
	.quad	.LC5
	.byte	0x9f
	.quad	0
	.quad	0
.LLST26:
	.quad	.LVL66
	.quad	.LVL67
	.value	0xa
	.byte	0x3
	.quad	.LC6
	.byte	0x9f
	.quad	0
	.quad	0
.LLST29:
	.quad	.LVL70
	.quad	.LVL71-1
	.value	0x9
	.byte	0x3
	.quad	A
	.quad	0
	.quad	0
.LLST30:
	.quad	.LVL71
	.quad	.LVL72-1
	.value	0x9
	.byte	0x3
	.quad	B
	.quad	0
	.quad	0
.LLST31:
	.quad	.LVL72
	.quad	.LVL73-1
	.value	0x9
	.byte	0x3
	.quad	C1
	.quad	0
	.quad	0
.LLST32:
	.quad	.LVL73
	.quad	.LVL74-1
	.value	0x9
	.byte	0x3
	.quad	C2
	.quad	0
	.quad	0
.LLST33:
	.quad	.LVL74
	.quad	.LVL75-1
	.value	0x9
	.byte	0x3
	.quad	C3
	.quad	0
	.quad	0
.LLST11:
	.quad	.LVL29
	.quad	.LVL30
	.value	0x2
	.byte	0x30
	.byte	0x9f
	.quad	0
	.quad	0
.LLST12:
	.quad	.LVL31
	.quad	.LVL35
	.value	0x1
	.byte	0x51
	.quad	.LVL35
	.quad	.LVL38
	.value	0x3
	.byte	0x71
	.sleb128 -64
	.byte	0x9f
	.quad	.LVL40
	.quad	.LVL41
	.value	0x1
	.byte	0x51
	.quad	.LVL41
	.quad	.LVL42
	.value	0x1
	.byte	0x52
	.quad	.LVL42
	.quad	.LFE4794
	.value	0x1
	.byte	0x51
	.quad	0
	.quad	0
.LLST13:
	.quad	.LVL31
	.quad	.LVL32
	.value	0x4
	.byte	0xa
	.value	0x400
	.byte	0x9f
	.quad	0
	.quad	0
.LLST14:
	.quad	.LVL33
	.quad	.LVL34
	.value	0x1
	.byte	0x58
	.quad	.LVL34
	.quad	.LVL36
	.value	0x1
	.byte	0x50
	.quad	.LVL36
	.quad	.LVL38
	.value	0x3
	.byte	0x70
	.sleb128 -64
	.byte	0x9f
	.quad	.LVL40
	.quad	.LFE4794
	.value	0x1
	.byte	0x50
	.quad	0
	.quad	0
.LLST16:
	.quad	.LVL33
	.quad	.LVL34
	.value	0x3
	.byte	0x8
	.byte	0x80
	.byte	0x9f
	.quad	0
	.quad	0
.LLST17:
	.quad	.LVL34
	.quad	.LVL37
	.value	0x1
	.byte	0x61
	.quad	0
	.quad	0
.LLST18:
	.quad	.LVL38
	.quad	.LVL39
	.value	0x1
	.byte	0x61
	.quad	0
	.quad	0
.LLST15:
	.quad	.LVL32
	.quad	.LVL33
	.value	0x1
	.byte	0x54
	.quad	0
	.quad	0
.LLST8:
	.quad	.LVL28
	.quad	.LVL29
	.value	0x4
	.byte	0x40
	.byte	0x43
	.byte	0x24
	.byte	0x9f
	.quad	0
	.quad	0
.LLST9:
	.quad	.LVL28
	.quad	.LVL29
	.value	0x2
	.byte	0x30
	.byte	0x9f
	.quad	0
	.quad	0
.LLST10:
	.quad	.LVL28
	.quad	.LVL29-1
	.value	0x9
	.byte	0x3
	.quad	C3
	.quad	0
	.quad	0
.LLST3:
	.quad	.LVL8
	.quad	.LVL9
	.value	0x2
	.byte	0x30
	.byte	0x9f
	.quad	0
	.quad	0
.LLST4:
	.quad	.LVL9
	.quad	.LVL22
	.value	0x1
	.byte	0x5d
	.quad	.LVL22
	.quad	.LVL23
	.value	0x4
	.byte	0x7d
	.sleb128 -8192
	.byte	0x9f
	.quad	.LVL24
	.quad	.LFE4793
	.value	0x1
	.byte	0x5d
	.quad	0
	.quad	0
.LLST5:
	.quad	.LVL10
	.quad	.LVL21
	.value	0xe
	.byte	0x3
	.quad	B
	.byte	0x6
	.byte	0x7b
	.sleb128 0
	.byte	0x22
	.byte	0x9f
	.quad	.LVL21
	.quad	.LVL24
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
	.quad	.LVL24
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
	.quad	.LVL11
	.quad	.LVL19
	.value	0x1
	.byte	0x62
	.quad	.LVL24
	.quad	.LFE4793
	.value	0x1
	.byte	0x62
	.quad	0
	.quad	0
.LLST7:
	.quad	.LVL11
	.quad	.LVL12
	.value	0x2
	.byte	0x30
	.byte	0x9f
	.quad	.LVL12
	.quad	.LVL13
	.value	0x2
	.byte	0x31
	.byte	0x9f
	.quad	.LVL13
	.quad	.LVL14
	.value	0x2
	.byte	0x32
	.byte	0x9f
	.quad	.LVL15
	.quad	.LVL16
	.value	0x1
	.byte	0x5a
	.quad	.LVL16
	.quad	.LVL17
	.value	0x1
	.byte	0x50
	.quad	.LVL17
	.quad	.LVL18
	.value	0x3
	.byte	0x7a
	.sleb128 -1
	.byte	0x9f
	.quad	.LVL24
	.quad	.LVL25
	.value	0x2
	.byte	0x31
	.byte	0x9f
	.quad	.LVL25
	.quad	.LVL26
	.value	0x2
	.byte	0x30
	.byte	0x9f
	.quad	.LVL27
	.quad	.LFE4793
	.value	0x2
	.byte	0x32
	.byte	0x9f
	.quad	0
	.quad	0
.LLST0:
	.quad	.LVL3
	.quad	.LVL4
	.value	0x2
	.byte	0x30
	.byte	0x9f
	.quad	0
	.quad	0
.LLST1:
	.quad	.LVL4
	.quad	.LVL5
	.value	0x2
	.byte	0x30
	.byte	0x9f
	.quad	0
	.quad	0
.LLST2:
	.quad	.LVL6
	.quad	.LVL7
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
	.quad	.LBB59
	.quad	.LBE59
	.quad	.LBB64
	.quad	.LBE64
	.quad	0
	.quad	0
	.quad	.LBB60
	.quad	.LBE60
	.quad	.LBB63
	.quad	.LBE63
	.quad	0
	.quad	0
	.quad	.LBB61
	.quad	.LBE61
	.quad	.LBB62
	.quad	.LBE62
	.quad	0
	.quad	0
	.quad	.LBB65
	.quad	.LBE65
	.quad	.LBB80
	.quad	.LBE80
	.quad	0
	.quad	0
	.quad	.LBB66
	.quad	.LBE66
	.quad	.LBB79
	.quad	.LBE79
	.quad	0
	.quad	0
	.quad	.LBB67
	.quad	.LBE67
	.quad	.LBB76
	.quad	.LBE76
	.quad	.LBB77
	.quad	.LBE77
	.quad	.LBB78
	.quad	.LBE78
	.quad	0
	.quad	0
	.quad	.LBB68
	.quad	.LBE68
	.quad	.LBB73
	.quad	.LBE73
	.quad	.LBB74
	.quad	.LBE74
	.quad	.LBB75
	.quad	.LBE75
	.quad	0
	.quad	0
	.quad	.LBB69
	.quad	.LBE69
	.quad	.LBB70
	.quad	.LBE70
	.quad	.LBB71
	.quad	.LBE71
	.quad	.LBB72
	.quad	.LBE72
	.quad	0
	.quad	0
	.quad	.LBB81
	.quad	.LBE81
	.quad	.LBB84
	.quad	.LBE84
	.quad	0
	.quad	0
	.quad	.LBB102
	.quad	.LBE102
	.quad	.LBB106
	.quad	.LBE106
	.quad	.LBB107
	.quad	.LBE107
	.quad	0
	.quad	0
	.quad	.LBB108
	.quad	.LBE108
	.quad	.LBB113
	.quad	.LBE113
	.quad	.LBB114
	.quad	.LBE114
	.quad	.LBB115
	.quad	.LBE115
	.quad	0
	.quad	0
	.quad	.LBB116
	.quad	.LBE116
	.quad	.LBB121
	.quad	.LBE121
	.quad	.LBB122
	.quad	.LBE122
	.quad	.LBB123
	.quad	.LBE123
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
.LASF98:
	.string	"drand48"
.LASF68:
	.string	"__morecore"
.LASF33:
	.string	"_shortbuf"
.LASF109:
	.string	"_IO_lock_t"
.LASF53:
	.string	"stderr"
.LASF22:
	.string	"_IO_buf_end"
.LASF112:
	.string	"_mm256_store_pd"
.LASF20:
	.string	"_IO_write_end"
.LASF4:
	.string	"unsigned int"
.LASF70:
	.string	"__malloc_hook"
.LASF14:
	.string	"_flags"
.LASF26:
	.string	"_markers"
.LASF107:
	.string	"mulmatrix.c"
.LASF87:
	.string	"_mm256_load_pd"
.LASF61:
	.string	"__timezone"
.LASF47:
	.string	"_pos"
.LASF52:
	.string	"stdout"
.LASF25:
	.string	"_IO_save_end"
.LASF108:
	.string	"/mnt/c/Users/Nika/Documents/GitHub/790541_AVP"
.LASF106:
	.string	"GNU C11 7.4.0 -mavx -mfma -mtune=generic -march=x86-64 -g -O3 -fstack-protector-strong"
.LASF88:
	.string	"_mm256_broadcast_sd"
.LASF95:
	.string	"__len"
.LASF76:
	.string	"__v4df"
.LASF74:
	.string	"long long unsigned int"
.LASF55:
	.string	"sys_errlist"
.LASF24:
	.string	"_IO_backup_base"
.LASF35:
	.string	"_offset"
.LASF54:
	.string	"sys_nerr"
.LASF93:
	.string	"__dest"
.LASF72:
	.string	"__memalign_hook"
.LASF28:
	.string	"_fileno"
.LASF102:
	.string	"puts"
.LASF58:
	.string	"tv_nsec"
.LASF8:
	.string	"size_t"
.LASF86:
	.string	"_mm256_fmadd_pd"
.LASF114:
	.string	"_mm_malloc"
.LASF17:
	.string	"_IO_read_base"
.LASF69:
	.string	"__free_hook"
.LASF78:
	.string	"argc"
.LASF51:
	.string	"stdin"
.LASF45:
	.string	"_next"
.LASF100:
	.string	"free"
.LASF84:
	.string	"mulstr"
.LASF97:
	.string	"__fmt"
.LASF113:
	.string	"_mm_free"
.LASF56:
	.string	"timespec"
.LASF13:
	.string	"char"
.LASF41:
	.string	"_mode"
.LASF60:
	.string	"__daylight"
.LASF62:
	.string	"tzname"
.LASF44:
	.string	"_IO_marker"
.LASF15:
	.string	"_IO_read_ptr"
.LASF103:
	.string	"__builtin_puts"
.LASF71:
	.string	"__realloc_hook"
.LASF77:
	.string	"__m256d"
.LASF18:
	.string	"_IO_write_base"
.LASF48:
	.string	"_IO_2_1_stdin_"
.LASF66:
	.string	"long long int"
.LASF96:
	.string	"printf"
.LASF49:
	.string	"_IO_2_1_stdout_"
.LASF23:
	.string	"_IO_save_base"
.LASF99:
	.string	"__printf_chk"
.LASF12:
	.string	"__syscall_slong_t"
.LASF92:
	.string	"memset"
.LASF101:
	.string	"clock_gettime"
.LASF65:
	.string	"ptrdiff_t"
.LASF83:
	.string	"mulvec"
.LASF36:
	.string	"__pad1"
.LASF37:
	.string	"__pad2"
.LASF38:
	.string	"__pad3"
.LASF39:
	.string	"__pad4"
.LASF40:
	.string	"__pad5"
.LASF32:
	.string	"_vtable_offset"
.LASF73:
	.string	"__after_morecore_hook"
.LASF79:
	.string	"argv"
.LASF67:
	.string	"long double"
.LASF64:
	.string	"timezone"
.LASF94:
	.string	"__ch"
.LASF16:
	.string	"_IO_read_end"
.LASF6:
	.string	"short int"
.LASF7:
	.string	"long int"
.LASF115:
	.string	"__stack_chk_fail"
.LASF110:
	.string	"_IO_FILE_plus"
.LASF85:
	.string	"mulmat"
.LASF104:
	.string	"__builtin_memset"
.LASF34:
	.string	"_lock"
.LASF57:
	.string	"tv_sec"
.LASF1:
	.string	"long unsigned int"
.LASF30:
	.string	"_old_offset"
.LASF43:
	.string	"_IO_FILE"
.LASF75:
	.string	"float"
.LASF89:
	.string	"__ptr"
.LASF91:
	.string	"__alignment"
.LASF2:
	.string	"unsigned char"
.LASF59:
	.string	"__tzname"
.LASF46:
	.string	"_sbuf"
.LASF80:
	.string	"tres1"
.LASF81:
	.string	"tres2"
.LASF82:
	.string	"tres3"
.LASF19:
	.string	"_IO_write_ptr"
.LASF11:
	.string	"__time_t"
.LASF63:
	.string	"daylight"
.LASF9:
	.string	"__off_t"
.LASF5:
	.string	"signed char"
.LASF3:
	.string	"short unsigned int"
.LASF111:
	.string	"main"
.LASF0:
	.string	"double"
.LASF27:
	.string	"_chain"
.LASF29:
	.string	"_flags2"
.LASF105:
	.string	"posix_memalign"
.LASF90:
	.string	"__size"
.LASF31:
	.string	"_cur_column"
.LASF50:
	.string	"_IO_2_1_stderr_"
.LASF10:
	.string	"__off64_t"
.LASF42:
	.string	"_unused2"
.LASF21:
	.string	"_IO_buf_base"
	.ident	"GCC: (Ubuntu 7.4.0-1ubuntu1~18.04.1) 7.4.0"
	.section	.note.GNU-stack,"",@progbits
