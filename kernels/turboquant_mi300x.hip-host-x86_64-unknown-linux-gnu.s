	.file	"turboquant_mi300x.hip.cpp"
	.text
	.globl	_Z38__device_stub__tqm_quantize_kernel_tq3PKfS0_P16block_tq3_mi300xi # -- Begin function _Z38__device_stub__tqm_quantize_kernel_tq3PKfS0_P16block_tq3_mi300xi
	.p2align	4
	.type	_Z38__device_stub__tqm_quantize_kernel_tq3PKfS0_P16block_tq3_mi300xi,@function
_Z38__device_stub__tqm_quantize_kernel_tq3PKfS0_P16block_tq3_mi300xi: # @_Z38__device_stub__tqm_quantize_kernel_tq3PKfS0_P16block_tq3_mi300xi
	.cfi_startproc
# %bb.0:
	subq	$120, %rsp
	.cfi_def_cfa_offset 128
	movq	%rdi, 72(%rsp)
	movq	%rsi, 64(%rsp)
	movq	%rdx, 56(%rsp)
	movl	%ecx, 4(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 80(%rsp)
	leaq	64(%rsp), %rax
	movq	%rax, 88(%rsp)
	leaq	56(%rsp), %rax
	movq	%rax, 96(%rsp)
	leaq	4(%rsp), %rax
	movq	%rax, 104(%rsp)
	leaq	40(%rsp), %rdi
	leaq	24(%rsp), %rsi
	leaq	16(%rsp), %rdx
	leaq	8(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	40(%rsp), %rsi
	movl	48(%rsp), %edx
	movq	24(%rsp), %rcx
	movl	32(%rsp), %r8d
	movq	_Z23tqm_quantize_kernel_tq3PKfS0_P16block_tq3_mi300xi@GOTPCREL(%rip), %rdi
	leaq	80(%rsp), %r9
	pushq	8(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$136, %rsp
	.cfi_adjust_cfa_offset -136
	retq
.Lfunc_end0:
	.size	_Z38__device_stub__tqm_quantize_kernel_tq3PKfS0_P16block_tq3_mi300xi, .Lfunc_end0-_Z38__device_stub__tqm_quantize_kernel_tq3PKfS0_P16block_tq3_mi300xi
	.cfi_endproc
                                        # -- End function
	.globl	_Z40__device_stub__tqm_dequantize_kernel_tq3PK16block_tq3_mi300xPKfPfi # -- Begin function _Z40__device_stub__tqm_dequantize_kernel_tq3PK16block_tq3_mi300xPKfPfi
	.p2align	4
	.type	_Z40__device_stub__tqm_dequantize_kernel_tq3PK16block_tq3_mi300xPKfPfi,@function
_Z40__device_stub__tqm_dequantize_kernel_tq3PK16block_tq3_mi300xPKfPfi: # @_Z40__device_stub__tqm_dequantize_kernel_tq3PK16block_tq3_mi300xPKfPfi
	.cfi_startproc
# %bb.0:
	subq	$120, %rsp
	.cfi_def_cfa_offset 128
	movq	%rdi, 72(%rsp)
	movq	%rsi, 64(%rsp)
	movq	%rdx, 56(%rsp)
	movl	%ecx, 4(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 80(%rsp)
	leaq	64(%rsp), %rax
	movq	%rax, 88(%rsp)
	leaq	56(%rsp), %rax
	movq	%rax, 96(%rsp)
	leaq	4(%rsp), %rax
	movq	%rax, 104(%rsp)
	leaq	40(%rsp), %rdi
	leaq	24(%rsp), %rsi
	leaq	16(%rsp), %rdx
	leaq	8(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	40(%rsp), %rsi
	movl	48(%rsp), %edx
	movq	24(%rsp), %rcx
	movl	32(%rsp), %r8d
	movq	_Z25tqm_dequantize_kernel_tq3PK16block_tq3_mi300xPKfPfi@GOTPCREL(%rip), %rdi
	leaq	80(%rsp), %r9
	pushq	8(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$136, %rsp
	.cfi_adjust_cfa_offset -136
	retq
.Lfunc_end1:
	.size	_Z40__device_stub__tqm_dequantize_kernel_tq3PK16block_tq3_mi300xPKfPfi, .Lfunc_end1-_Z40__device_stub__tqm_dequantize_kernel_tq3PK16block_tq3_mi300xPKfPfi
	.cfi_endproc
                                        # -- End function
	.globl	_Z39__device_stub__tqm_fused_dot_kernel_tq3PKfPK16block_tq3_mi300xPfii # -- Begin function _Z39__device_stub__tqm_fused_dot_kernel_tq3PKfPK16block_tq3_mi300xPfii
	.p2align	4
	.type	_Z39__device_stub__tqm_fused_dot_kernel_tq3PKfPK16block_tq3_mi300xPfii,@function
_Z39__device_stub__tqm_fused_dot_kernel_tq3PKfPK16block_tq3_mi300xPfii: # @_Z39__device_stub__tqm_fused_dot_kernel_tq3PKfPK16block_tq3_mi300xPfii
	.cfi_startproc
# %bb.0:
	subq	$120, %rsp
	.cfi_def_cfa_offset 128
	movq	%rdi, 72(%rsp)
	movq	%rsi, 64(%rsp)
	movq	%rdx, 56(%rsp)
	movl	%ecx, 4(%rsp)
	movl	%r8d, (%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 80(%rsp)
	leaq	64(%rsp), %rax
	movq	%rax, 88(%rsp)
	leaq	56(%rsp), %rax
	movq	%rax, 96(%rsp)
	leaq	4(%rsp), %rax
	movq	%rax, 104(%rsp)
	movq	%rsp, %rax
	movq	%rax, 112(%rsp)
	leaq	40(%rsp), %rdi
	leaq	24(%rsp), %rsi
	leaq	16(%rsp), %rdx
	leaq	8(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	40(%rsp), %rsi
	movl	48(%rsp), %edx
	movq	24(%rsp), %rcx
	movl	32(%rsp), %r8d
	movq	_Z24tqm_fused_dot_kernel_tq3PKfPK16block_tq3_mi300xPfii@GOTPCREL(%rip), %rdi
	leaq	80(%rsp), %r9
	pushq	8(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$136, %rsp
	.cfi_adjust_cfa_offset -136
	retq
.Lfunc_end2:
	.size	_Z39__device_stub__tqm_fused_dot_kernel_tq3PKfPK16block_tq3_mi300xPfii, .Lfunc_end2-_Z39__device_stub__tqm_fused_dot_kernel_tq3PKfPK16block_tq3_mi300xPfii
	.cfi_endproc
                                        # -- End function
	.globl	_Z38__device_stub__tqm_quantize_kernel_tq4PKfS0_P16block_tq4_mi300xi # -- Begin function _Z38__device_stub__tqm_quantize_kernel_tq4PKfS0_P16block_tq4_mi300xi
	.p2align	4
	.type	_Z38__device_stub__tqm_quantize_kernel_tq4PKfS0_P16block_tq4_mi300xi,@function
_Z38__device_stub__tqm_quantize_kernel_tq4PKfS0_P16block_tq4_mi300xi: # @_Z38__device_stub__tqm_quantize_kernel_tq4PKfS0_P16block_tq4_mi300xi
	.cfi_startproc
# %bb.0:
	subq	$120, %rsp
	.cfi_def_cfa_offset 128
	movq	%rdi, 72(%rsp)
	movq	%rsi, 64(%rsp)
	movq	%rdx, 56(%rsp)
	movl	%ecx, 4(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 80(%rsp)
	leaq	64(%rsp), %rax
	movq	%rax, 88(%rsp)
	leaq	56(%rsp), %rax
	movq	%rax, 96(%rsp)
	leaq	4(%rsp), %rax
	movq	%rax, 104(%rsp)
	leaq	40(%rsp), %rdi
	leaq	24(%rsp), %rsi
	leaq	16(%rsp), %rdx
	leaq	8(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	40(%rsp), %rsi
	movl	48(%rsp), %edx
	movq	24(%rsp), %rcx
	movl	32(%rsp), %r8d
	movq	_Z23tqm_quantize_kernel_tq4PKfS0_P16block_tq4_mi300xi@GOTPCREL(%rip), %rdi
	leaq	80(%rsp), %r9
	pushq	8(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$136, %rsp
	.cfi_adjust_cfa_offset -136
	retq
.Lfunc_end3:
	.size	_Z38__device_stub__tqm_quantize_kernel_tq4PKfS0_P16block_tq4_mi300xi, .Lfunc_end3-_Z38__device_stub__tqm_quantize_kernel_tq4PKfS0_P16block_tq4_mi300xi
	.cfi_endproc
                                        # -- End function
	.globl	_Z40__device_stub__tqm_dequantize_kernel_tq4PK16block_tq4_mi300xPKfPfi # -- Begin function _Z40__device_stub__tqm_dequantize_kernel_tq4PK16block_tq4_mi300xPKfPfi
	.p2align	4
	.type	_Z40__device_stub__tqm_dequantize_kernel_tq4PK16block_tq4_mi300xPKfPfi,@function
_Z40__device_stub__tqm_dequantize_kernel_tq4PK16block_tq4_mi300xPKfPfi: # @_Z40__device_stub__tqm_dequantize_kernel_tq4PK16block_tq4_mi300xPKfPfi
	.cfi_startproc
# %bb.0:
	subq	$120, %rsp
	.cfi_def_cfa_offset 128
	movq	%rdi, 72(%rsp)
	movq	%rsi, 64(%rsp)
	movq	%rdx, 56(%rsp)
	movl	%ecx, 4(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 80(%rsp)
	leaq	64(%rsp), %rax
	movq	%rax, 88(%rsp)
	leaq	56(%rsp), %rax
	movq	%rax, 96(%rsp)
	leaq	4(%rsp), %rax
	movq	%rax, 104(%rsp)
	leaq	40(%rsp), %rdi
	leaq	24(%rsp), %rsi
	leaq	16(%rsp), %rdx
	leaq	8(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	40(%rsp), %rsi
	movl	48(%rsp), %edx
	movq	24(%rsp), %rcx
	movl	32(%rsp), %r8d
	movq	_Z25tqm_dequantize_kernel_tq4PK16block_tq4_mi300xPKfPfi@GOTPCREL(%rip), %rdi
	leaq	80(%rsp), %r9
	pushq	8(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$136, %rsp
	.cfi_adjust_cfa_offset -136
	retq
.Lfunc_end4:
	.size	_Z40__device_stub__tqm_dequantize_kernel_tq4PK16block_tq4_mi300xPKfPfi, .Lfunc_end4-_Z40__device_stub__tqm_dequantize_kernel_tq4PK16block_tq4_mi300xPKfPfi
	.cfi_endproc
                                        # -- End function
	.globl	_Z29__device_stub__tqm_qjl_kernelPKfPK16block_tq3_mi300xS0_S0_P16block_qjl_mi300xi # -- Begin function _Z29__device_stub__tqm_qjl_kernelPKfPK16block_tq3_mi300xS0_S0_P16block_qjl_mi300xi
	.p2align	4
	.type	_Z29__device_stub__tqm_qjl_kernelPKfPK16block_tq3_mi300xS0_S0_P16block_qjl_mi300xi,@function
_Z29__device_stub__tqm_qjl_kernelPKfPK16block_tq3_mi300xS0_S0_P16block_qjl_mi300xi: # @_Z29__device_stub__tqm_qjl_kernelPKfPK16block_tq3_mi300xS0_S0_P16block_qjl_mi300xi
	.cfi_startproc
# %bb.0:
	subq	$152, %rsp
	.cfi_def_cfa_offset 160
	movq	%rdi, 88(%rsp)
	movq	%rsi, 80(%rsp)
	movq	%rdx, 72(%rsp)
	movq	%rcx, 64(%rsp)
	movq	%r8, 56(%rsp)
	movl	%r9d, 4(%rsp)
	leaq	88(%rsp), %rax
	movq	%rax, 96(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 104(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 112(%rsp)
	leaq	64(%rsp), %rax
	movq	%rax, 120(%rsp)
	leaq	56(%rsp), %rax
	movq	%rax, 128(%rsp)
	leaq	4(%rsp), %rax
	movq	%rax, 136(%rsp)
	leaq	40(%rsp), %rdi
	leaq	24(%rsp), %rsi
	leaq	16(%rsp), %rdx
	leaq	8(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	40(%rsp), %rsi
	movl	48(%rsp), %edx
	movq	24(%rsp), %rcx
	movl	32(%rsp), %r8d
	movq	_Z14tqm_qjl_kernelPKfPK16block_tq3_mi300xS0_S0_P16block_qjl_mi300xi@GOTPCREL(%rip), %rdi
	leaq	96(%rsp), %r9
	pushq	8(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$168, %rsp
	.cfi_adjust_cfa_offset -168
	retq
.Lfunc_end5:
	.size	_Z29__device_stub__tqm_qjl_kernelPKfPK16block_tq3_mi300xS0_S0_P16block_qjl_mi300xi, .Lfunc_end5-_Z29__device_stub__tqm_qjl_kernelPKfPK16block_tq3_mi300xS0_S0_P16block_qjl_mi300xi
	.cfi_endproc
                                        # -- End function
	.globl	tqm_init                        # -- Begin function tqm_init
	.p2align	4
	.type	tqm_init,@function
tqm_init:                               # @tqm_init
	.cfi_startproc
# %bb.0:
	pushq	%rax
	.cfi_def_cfa_offset 16
	leaq	d_cb3(%rip), %rdi
	leaq	_ZL19TQ3_CODEBOOK_MI300X(%rip), %rsi
	movl	$32, %edx
	xorl	%ecx, %ecx
	movl	$1, %r8d
	callq	hipMemcpyToSymbol@PLT
	leaq	d_cb4(%rip), %rdi
	leaq	_ZL19TQ4_CODEBOOK_MI300X(%rip), %rsi
	movl	$64, %edx
	xorl	%ecx, %ecx
	movl	$1, %r8d
	callq	hipMemcpyToSymbol@PLT
	leaq	d_cb2(%rip), %rdi
	leaq	_ZL19TQ2_CODEBOOK_MI300X(%rip), %rsi
	movl	$16, %edx
	xorl	%ecx, %ecx
	movl	$1, %r8d
	popq	%rax
	.cfi_def_cfa_offset 8
	jmp	hipMemcpyToSymbol@PLT           # TAILCALL
.Lfunc_end6:
	.size	tqm_init, .Lfunc_end6-tqm_init
	.cfi_endproc
                                        # -- End function
	.globl	tqm_quantize_tq3                # -- Begin function tqm_quantize_tq3
	.p2align	4
	.type	tqm_quantize_tq3,@function
tqm_quantize_tq3:                       # @tqm_quantize_tq3
	.cfi_startproc
# %bb.0:
	testl	%ecx, %ecx
	jle	.LBB7_4
# %bb.1:
	pushq	%r15
	.cfi_def_cfa_offset 16
	pushq	%r14
	.cfi_def_cfa_offset 24
	pushq	%r12
	.cfi_def_cfa_offset 32
	pushq	%rbx
	.cfi_def_cfa_offset 40
	subq	$120, %rsp
	.cfi_def_cfa_offset 160
	.cfi_offset %rbx, -40
	.cfi_offset %r12, -32
	.cfi_offset %r14, -24
	.cfi_offset %r15, -16
	movq	%r8, %r9
	movl	%ecx, %ebx
	movq	%rdx, %r14
	movq	%rsi, %r15
	movq	%rdi, %r12
	movl	%ecx, %edi
	movabsq	$4294967296, %rdx               # imm = 0x100000000
	orq	%rdx, %rdi
	orq	$128, %rdx
	movl	$1, %esi
	movl	$1, %ecx
	xorl	%r8d, %r8d
	callq	__hipPushCallConfiguration@PLT
	testl	%eax, %eax
	jne	.LBB7_3
# %bb.2:
	movq	%r12, 72(%rsp)
	movq	%r15, 64(%rsp)
	movq	%r14, 56(%rsp)
	movl	%ebx, 4(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 80(%rsp)
	leaq	64(%rsp), %rax
	movq	%rax, 88(%rsp)
	leaq	56(%rsp), %rax
	movq	%rax, 96(%rsp)
	leaq	4(%rsp), %rax
	movq	%rax, 104(%rsp)
	leaq	40(%rsp), %rdi
	leaq	24(%rsp), %rsi
	leaq	16(%rsp), %rdx
	leaq	8(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	40(%rsp), %rsi
	movl	48(%rsp), %edx
	movq	24(%rsp), %rcx
	movl	32(%rsp), %r8d
	movq	_Z23tqm_quantize_kernel_tq3PKfS0_P16block_tq3_mi300xi@GOTPCREL(%rip), %rdi
	leaq	80(%rsp), %r9
	pushq	8(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.LBB7_3:
	addq	$120, %rsp
	.cfi_def_cfa_offset 40
	popq	%rbx
	.cfi_def_cfa_offset 32
	popq	%r12
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	.cfi_restore %rbx
	.cfi_restore %r12
	.cfi_restore %r14
	.cfi_restore %r15
.LBB7_4:
	retq
.Lfunc_end7:
	.size	tqm_quantize_tq3, .Lfunc_end7-tqm_quantize_tq3
	.cfi_endproc
                                        # -- End function
	.globl	tqm_dequantize_tq3              # -- Begin function tqm_dequantize_tq3
	.p2align	4
	.type	tqm_dequantize_tq3,@function
tqm_dequantize_tq3:                     # @tqm_dequantize_tq3
	.cfi_startproc
# %bb.0:
	testl	%ecx, %ecx
	jle	.LBB8_4
# %bb.1:
	pushq	%r15
	.cfi_def_cfa_offset 16
	pushq	%r14
	.cfi_def_cfa_offset 24
	pushq	%r12
	.cfi_def_cfa_offset 32
	pushq	%rbx
	.cfi_def_cfa_offset 40
	subq	$120, %rsp
	.cfi_def_cfa_offset 160
	.cfi_offset %rbx, -40
	.cfi_offset %r12, -32
	.cfi_offset %r14, -24
	.cfi_offset %r15, -16
	movq	%r8, %r9
	movl	%ecx, %ebx
	movq	%rdx, %r14
	movq	%rsi, %r15
	movq	%rdi, %r12
	movl	%ecx, %edi
	movabsq	$4294967296, %rdx               # imm = 0x100000000
	orq	%rdx, %rdi
	orq	$128, %rdx
	movl	$1, %esi
	movl	$1, %ecx
	xorl	%r8d, %r8d
	callq	__hipPushCallConfiguration@PLT
	testl	%eax, %eax
	jne	.LBB8_3
# %bb.2:
	movq	%r12, 72(%rsp)
	movq	%r15, 64(%rsp)
	movq	%r14, 56(%rsp)
	movl	%ebx, 4(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 80(%rsp)
	leaq	64(%rsp), %rax
	movq	%rax, 88(%rsp)
	leaq	56(%rsp), %rax
	movq	%rax, 96(%rsp)
	leaq	4(%rsp), %rax
	movq	%rax, 104(%rsp)
	leaq	40(%rsp), %rdi
	leaq	24(%rsp), %rsi
	leaq	16(%rsp), %rdx
	leaq	8(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	40(%rsp), %rsi
	movl	48(%rsp), %edx
	movq	24(%rsp), %rcx
	movl	32(%rsp), %r8d
	movq	_Z25tqm_dequantize_kernel_tq3PK16block_tq3_mi300xPKfPfi@GOTPCREL(%rip), %rdi
	leaq	80(%rsp), %r9
	pushq	8(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.LBB8_3:
	addq	$120, %rsp
	.cfi_def_cfa_offset 40
	popq	%rbx
	.cfi_def_cfa_offset 32
	popq	%r12
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	.cfi_restore %rbx
	.cfi_restore %r12
	.cfi_restore %r14
	.cfi_restore %r15
.LBB8_4:
	retq
.Lfunc_end8:
	.size	tqm_dequantize_tq3, .Lfunc_end8-tqm_dequantize_tq3
	.cfi_endproc
                                        # -- End function
	.globl	tqm_fused_dot_tq3               # -- Begin function tqm_fused_dot_tq3
	.p2align	4
	.type	tqm_fused_dot_tq3,@function
tqm_fused_dot_tq3:                      # @tqm_fused_dot_tq3
	.cfi_startproc
# %bb.0:
	pushq	%r15
	.cfi_def_cfa_offset 16
	pushq	%r14
	.cfi_def_cfa_offset 24
	pushq	%r13
	.cfi_def_cfa_offset 32
	pushq	%r12
	.cfi_def_cfa_offset 40
	pushq	%rbx
	.cfi_def_cfa_offset 48
	subq	$128, %rsp
	.cfi_def_cfa_offset 176
	.cfi_offset %rbx, -48
	.cfi_offset %r12, -40
	.cfi_offset %r13, -32
	.cfi_offset %r14, -24
	.cfi_offset %r15, -16
	movl	%ecx, %r14d
	testl	%ecx, %ecx
	setle	%al
	testl	%r8d, %r8d
	setle	%cl
	orb	%al, %cl
	jne	.LBB9_3
# %bb.1:
	movl	%r8d, %ebx
	movq	%rdx, %r15
	movq	%rsi, %r12
	movq	%rdi, %r13
	movl	%r8d, %eax
	movq	%r14, %rdi
	shlq	$32, %rdi
	orq	%rax, %rdi
	movabsq	$4294967424, %rdx               # imm = 0x100000080
	movl	$1, %esi
	movl	$1, %ecx
	xorl	%r8d, %r8d
	callq	__hipPushCallConfiguration@PLT
	testl	%eax, %eax
	jne	.LBB9_3
# %bb.2:
	movq	%r13, 72(%rsp)
	movq	%r12, 64(%rsp)
	movq	%r15, 56(%rsp)
	movl	%r14d, 4(%rsp)
	movl	%ebx, (%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 80(%rsp)
	leaq	64(%rsp), %rax
	movq	%rax, 88(%rsp)
	leaq	56(%rsp), %rax
	movq	%rax, 96(%rsp)
	leaq	4(%rsp), %rax
	movq	%rax, 104(%rsp)
	movq	%rsp, %rax
	movq	%rax, 112(%rsp)
	leaq	40(%rsp), %rdi
	leaq	24(%rsp), %rsi
	leaq	16(%rsp), %rdx
	leaq	8(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	40(%rsp), %rsi
	movl	48(%rsp), %edx
	movq	24(%rsp), %rcx
	movl	32(%rsp), %r8d
	movq	_Z24tqm_fused_dot_kernel_tq3PKfPK16block_tq3_mi300xPfii@GOTPCREL(%rip), %rdi
	leaq	80(%rsp), %r9
	pushq	8(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.LBB9_3:
	addq	$128, %rsp
	.cfi_def_cfa_offset 48
	popq	%rbx
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end9:
	.size	tqm_fused_dot_tq3, .Lfunc_end9-tqm_fused_dot_tq3
	.cfi_endproc
                                        # -- End function
	.globl	tqm_quantize_tq4                # -- Begin function tqm_quantize_tq4
	.p2align	4
	.type	tqm_quantize_tq4,@function
tqm_quantize_tq4:                       # @tqm_quantize_tq4
	.cfi_startproc
# %bb.0:
	testl	%ecx, %ecx
	jle	.LBB10_4
# %bb.1:
	pushq	%r15
	.cfi_def_cfa_offset 16
	pushq	%r14
	.cfi_def_cfa_offset 24
	pushq	%r12
	.cfi_def_cfa_offset 32
	pushq	%rbx
	.cfi_def_cfa_offset 40
	subq	$120, %rsp
	.cfi_def_cfa_offset 160
	.cfi_offset %rbx, -40
	.cfi_offset %r12, -32
	.cfi_offset %r14, -24
	.cfi_offset %r15, -16
	movq	%r8, %r9
	movl	%ecx, %ebx
	movq	%rdx, %r14
	movq	%rsi, %r15
	movq	%rdi, %r12
	movl	%ecx, %edi
	movabsq	$4294967296, %rdx               # imm = 0x100000000
	orq	%rdx, %rdi
	orq	$128, %rdx
	movl	$1, %esi
	movl	$1, %ecx
	xorl	%r8d, %r8d
	callq	__hipPushCallConfiguration@PLT
	testl	%eax, %eax
	jne	.LBB10_3
# %bb.2:
	movq	%r12, 72(%rsp)
	movq	%r15, 64(%rsp)
	movq	%r14, 56(%rsp)
	movl	%ebx, 4(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 80(%rsp)
	leaq	64(%rsp), %rax
	movq	%rax, 88(%rsp)
	leaq	56(%rsp), %rax
	movq	%rax, 96(%rsp)
	leaq	4(%rsp), %rax
	movq	%rax, 104(%rsp)
	leaq	40(%rsp), %rdi
	leaq	24(%rsp), %rsi
	leaq	16(%rsp), %rdx
	leaq	8(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	40(%rsp), %rsi
	movl	48(%rsp), %edx
	movq	24(%rsp), %rcx
	movl	32(%rsp), %r8d
	movq	_Z23tqm_quantize_kernel_tq4PKfS0_P16block_tq4_mi300xi@GOTPCREL(%rip), %rdi
	leaq	80(%rsp), %r9
	pushq	8(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.LBB10_3:
	addq	$120, %rsp
	.cfi_def_cfa_offset 40
	popq	%rbx
	.cfi_def_cfa_offset 32
	popq	%r12
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	.cfi_restore %rbx
	.cfi_restore %r12
	.cfi_restore %r14
	.cfi_restore %r15
.LBB10_4:
	retq
.Lfunc_end10:
	.size	tqm_quantize_tq4, .Lfunc_end10-tqm_quantize_tq4
	.cfi_endproc
                                        # -- End function
	.globl	tqm_dequantize_tq4              # -- Begin function tqm_dequantize_tq4
	.p2align	4
	.type	tqm_dequantize_tq4,@function
tqm_dequantize_tq4:                     # @tqm_dequantize_tq4
	.cfi_startproc
# %bb.0:
	testl	%ecx, %ecx
	jle	.LBB11_4
# %bb.1:
	pushq	%r15
	.cfi_def_cfa_offset 16
	pushq	%r14
	.cfi_def_cfa_offset 24
	pushq	%r12
	.cfi_def_cfa_offset 32
	pushq	%rbx
	.cfi_def_cfa_offset 40
	subq	$120, %rsp
	.cfi_def_cfa_offset 160
	.cfi_offset %rbx, -40
	.cfi_offset %r12, -32
	.cfi_offset %r14, -24
	.cfi_offset %r15, -16
	movq	%r8, %r9
	movl	%ecx, %ebx
	movq	%rdx, %r14
	movq	%rsi, %r15
	movq	%rdi, %r12
	movl	%ecx, %edi
	movabsq	$4294967296, %rdx               # imm = 0x100000000
	orq	%rdx, %rdi
	orq	$128, %rdx
	movl	$1, %esi
	movl	$1, %ecx
	xorl	%r8d, %r8d
	callq	__hipPushCallConfiguration@PLT
	testl	%eax, %eax
	jne	.LBB11_3
# %bb.2:
	movq	%r12, 72(%rsp)
	movq	%r15, 64(%rsp)
	movq	%r14, 56(%rsp)
	movl	%ebx, 4(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 80(%rsp)
	leaq	64(%rsp), %rax
	movq	%rax, 88(%rsp)
	leaq	56(%rsp), %rax
	movq	%rax, 96(%rsp)
	leaq	4(%rsp), %rax
	movq	%rax, 104(%rsp)
	leaq	40(%rsp), %rdi
	leaq	24(%rsp), %rsi
	leaq	16(%rsp), %rdx
	leaq	8(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	40(%rsp), %rsi
	movl	48(%rsp), %edx
	movq	24(%rsp), %rcx
	movl	32(%rsp), %r8d
	movq	_Z25tqm_dequantize_kernel_tq4PK16block_tq4_mi300xPKfPfi@GOTPCREL(%rip), %rdi
	leaq	80(%rsp), %r9
	pushq	8(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.LBB11_3:
	addq	$120, %rsp
	.cfi_def_cfa_offset 40
	popq	%rbx
	.cfi_def_cfa_offset 32
	popq	%r12
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	.cfi_restore %rbx
	.cfi_restore %r12
	.cfi_restore %r14
	.cfi_restore %r15
.LBB11_4:
	retq
.Lfunc_end11:
	.size	tqm_dequantize_tq4, .Lfunc_end11-tqm_dequantize_tq4
	.cfi_endproc
                                        # -- End function
	.globl	tqm_qjl_keys                    # -- Begin function tqm_qjl_keys
	.p2align	4
	.type	tqm_qjl_keys,@function
tqm_qjl_keys:                           # @tqm_qjl_keys
	.cfi_startproc
# %bb.0:
	testl	%r9d, %r9d
	jle	.LBB12_4
# %bb.1:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	subq	$152, %rsp
	.cfi_def_cfa_offset 208
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movl	%r9d, %ebx
	movq	%r8, %r14
	movq	%rcx, %r15
	movq	%rdx, %r12
	movq	%rsi, %r13
	movq	%rdi, %rbp
	movq	208(%rsp), %r9
	movl	%ebx, %edi
	movabsq	$4294967296, %rdx               # imm = 0x100000000
	orq	%rdx, %rdi
	orq	$128, %rdx
	movl	$1, %esi
	movl	$1, %ecx
	xorl	%r8d, %r8d
	callq	__hipPushCallConfiguration@PLT
	testl	%eax, %eax
	jne	.LBB12_3
# %bb.2:
	movq	%rbp, 88(%rsp)
	movq	%r13, 80(%rsp)
	movq	%r12, 72(%rsp)
	movq	%r15, 64(%rsp)
	movq	%r14, 56(%rsp)
	movl	%ebx, 4(%rsp)
	leaq	88(%rsp), %rax
	movq	%rax, 96(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 104(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 112(%rsp)
	leaq	64(%rsp), %rax
	movq	%rax, 120(%rsp)
	leaq	56(%rsp), %rax
	movq	%rax, 128(%rsp)
	leaq	4(%rsp), %rax
	movq	%rax, 136(%rsp)
	leaq	40(%rsp), %rdi
	leaq	24(%rsp), %rsi
	leaq	16(%rsp), %rdx
	leaq	8(%rsp), %rcx
	callq	__hipPopCallConfiguration@PLT
	movq	40(%rsp), %rsi
	movl	48(%rsp), %edx
	movq	24(%rsp), %rcx
	movl	32(%rsp), %r8d
	movq	_Z14tqm_qjl_kernelPKfPK16block_tq3_mi300xS0_S0_P16block_qjl_mi300xi@GOTPCREL(%rip), %rdi
	leaq	96(%rsp), %r9
	pushq	8(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.LBB12_3:
	addq	$152, %rsp
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	.cfi_restore %rbx
	.cfi_restore %r12
	.cfi_restore %r13
	.cfi_restore %r14
	.cfi_restore %r15
	.cfi_restore %rbp
.LBB12_4:
	retq
.Lfunc_end12:
	.size	tqm_qjl_keys, .Lfunc_end12-tqm_qjl_keys
	.cfi_endproc
                                        # -- End function
	.section	.rodata.cst8,"aM",@progbits,8
	.p2align	3, 0x0                          # -- Begin function tqm_alloc_rotation
.LCPI13_0:
	.quad	0x3ca0000000000000              # double 1.1102230246251565E-16
.LCPI13_1:
	.quad	0x39b4484bfeebc2a0              # double 1.0000000000000001E-30
.LCPI13_2:
	.quad	0xc000000000000000              # double -2
.LCPI13_3:
	.quad	0x3cc921fb54442d18              # double 6.9757369960172635E-16
	.section	.rodata.cst4,"aM",@progbits,4
	.p2align	2, 0x0
.LCPI13_4:
	.long	0xc0400000                      # float -3
.LCPI13_5:
	.long	0xbf000000                      # float -0.5
	.text
	.globl	tqm_alloc_rotation
	.p2align	4
	.type	tqm_alloc_rotation,@function
tqm_alloc_rotation:                     # @tqm_alloc_rotation
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	subq	$552, %rsp                      # imm = 0x228
	.cfi_def_cfa_offset 608
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movq	%rsi, %rbx
	movq	%rdi, %r15
	movabsq	$6364136223846793005, %r12      # imm = 0x5851F42D4C957F2D
	movabsq	$1442695040888963407, %r13      # imm = 0x14057B7EF767814F
	movl	$65536, %edi                    # imm = 0x10000
	callq	malloc@PLT
	movq	%rax, %r14
	movabsq	$-2401053089206453570, %rax     # imm = 0xDEADBEEFCAFEBABE
	xorq	%r15, %rax
	xorl	%r15d, %r15d
	.p2align	4
.LBB13_1:                               # =>This Inner Loop Header: Depth=1
	imulq	%r12, %rax
	addq	%r13, %rax
	movq	%rax, %rbp
	imulq	%r12, %rbp
	addq	%r13, %rbp
	shrq	$11, %rax
	xorps	%xmm0, %xmm0
	cvtsi2sd	%rax, %xmm0
	mulsd	.LCPI13_0(%rip), %xmm0
	maxsd	.LCPI13_1(%rip), %xmm0
	movq	%rbp, %rax
	shrq	$11, %rax
	xorps	%xmm1, %xmm1
	cvtsi2sd	%rax, %xmm1
	movsd	%xmm1, 32(%rsp)                 # 8-byte Spill
	callq	log@PLT
	mulsd	.LCPI13_2(%rip), %xmm0
	sqrtsd	%xmm0, %xmm0
	movsd	%xmm0, 16(%rsp)                 # 8-byte Spill
	movsd	32(%rsp), %xmm0                 # 8-byte Reload
                                        # xmm0 = mem[0],zero
	mulsd	.LCPI13_3(%rip), %xmm0
	callq	cos@PLT
	mulsd	16(%rsp), %xmm0                 # 8-byte Folded Reload
	cvtsd2ss	%xmm0, %xmm0
	movss	%xmm0, (%r14,%r15,4)
	incq	%r15
	movq	%rbp, %rax
	cmpq	$16384, %r15                    # imm = 0x4000
	jne	.LBB13_1
# %bb.2:                                # %.preheader.preheader
	leaq	496(%r14), %rax
	xorl	%ecx, %ecx
	xorl	%edx, %edx
	jmp	.LBB13_3
	.p2align	4
.LBB13_6:                               # %vector.body
                                        #   in Loop: Header=BB13_3 Depth=1
	movups	496(%rsi), %xmm0
	mulps	%xmm0, %xmm0
	movups	464(%rsi), %xmm1
	mulps	%xmm1, %xmm1
	movups	432(%rsi), %xmm2
	mulps	%xmm2, %xmm2
	movups	400(%rsi), %xmm4
	mulps	%xmm4, %xmm4
	movups	368(%rsi), %xmm3
	mulps	%xmm3, %xmm3
	addps	%xmm4, %xmm3
	addps	%xmm2, %xmm3
	addps	%xmm1, %xmm3
	addps	%xmm0, %xmm3
	movups	336(%rsi), %xmm0
	mulps	%xmm0, %xmm0
	movups	304(%rsi), %xmm1
	mulps	%xmm1, %xmm1
	movups	272(%rsi), %xmm2
	mulps	%xmm2, %xmm2
	movups	240(%rsi), %xmm4
	mulps	%xmm4, %xmm4
	addps	%xmm2, %xmm4
	addps	%xmm1, %xmm4
	addps	%xmm0, %xmm4
	movups	208(%rsi), %xmm0
	mulps	%xmm0, %xmm0
	movups	176(%rsi), %xmm1
	mulps	%xmm1, %xmm1
	movups	144(%rsi), %xmm5
	mulps	%xmm5, %xmm5
	addps	%xmm1, %xmm5
	addps	%xmm0, %xmm5
	movups	112(%rsi), %xmm0
	mulps	%xmm0, %xmm0
	movups	80(%rsi), %xmm6
	mulps	%xmm6, %xmm6
	addps	%xmm0, %xmm6
	movups	(%rsi), %xmm1
	movups	16(%rsi), %xmm0
	movups	32(%rsi), %xmm2
	movups	48(%rsi), %xmm7
	mulps	%xmm7, %xmm7
	mulps	%xmm0, %xmm0
	addps	%xmm7, %xmm0
	addps	%xmm6, %xmm0
	addps	%xmm5, %xmm0
	addps	%xmm4, %xmm0
	addps	%xmm3, %xmm0
	movups	480(%rsi), %xmm4
	mulps	%xmm4, %xmm4
	movups	448(%rsi), %xmm5
	mulps	%xmm5, %xmm5
	movups	416(%rsi), %xmm6
	mulps	%xmm6, %xmm6
	movups	384(%rsi), %xmm7
	mulps	%xmm7, %xmm7
	movups	352(%rsi), %xmm3
	mulps	%xmm3, %xmm3
	addps	%xmm7, %xmm3
	addps	%xmm6, %xmm3
	addps	%xmm5, %xmm3
	addps	%xmm4, %xmm3
	movups	320(%rsi), %xmm5
	mulps	%xmm5, %xmm5
	movups	288(%rsi), %xmm6
	mulps	%xmm6, %xmm6
	movups	256(%rsi), %xmm7
	mulps	%xmm7, %xmm7
	movups	224(%rsi), %xmm4
	mulps	%xmm4, %xmm4
	addps	%xmm7, %xmm4
	addps	%xmm6, %xmm4
	addps	%xmm5, %xmm4
	movups	192(%rsi), %xmm5
	mulps	%xmm5, %xmm5
	movups	160(%rsi), %xmm6
	mulps	%xmm6, %xmm6
	movups	128(%rsi), %xmm7
	mulps	%xmm7, %xmm7
	addps	%xmm6, %xmm7
	addps	%xmm5, %xmm7
	movups	96(%rsi), %xmm5
	mulps	%xmm5, %xmm5
	movups	64(%rsi), %xmm6
	mulps	%xmm6, %xmm6
	addps	%xmm5, %xmm6
	mulps	%xmm2, %xmm2
	movaps	%xmm1, %xmm5
	mulps	%xmm1, %xmm5
	addps	%xmm2, %xmm5
	addps	%xmm6, %xmm5
	addps	%xmm7, %xmm5
	addps	%xmm4, %xmm5
	addps	%xmm3, %xmm5
	addps	%xmm0, %xmm5
	movaps	%xmm5, %xmm0
	unpckhpd	%xmm5, %xmm0                    # xmm0 = xmm0[1],xmm5[1]
	addps	%xmm5, %xmm0
	movaps	%xmm0, %xmm2
	shufps	$85, %xmm0, %xmm2               # xmm2 = xmm2[1,1],xmm0[1,1]
	addss	%xmm0, %xmm2
	xorps	%xmm0, %xmm0
	rsqrtss	%xmm2, %xmm0
	mulss	%xmm0, %xmm2
	mulss	%xmm0, %xmm2
	addss	.LCPI13_4(%rip), %xmm2
	mulss	.LCPI13_5(%rip), %xmm0
	mulss	%xmm2, %xmm0
	shufps	$0, %xmm0, %xmm0                # xmm0 = xmm0[0,0,0,0]
	mulps	%xmm0, %xmm1
	movups	%xmm1, (%rsi)
	movups	16(%rsi), %xmm1
	movups	32(%rsi), %xmm2
	movups	48(%rsi), %xmm3
	movups	64(%rsi), %xmm4
	mulps	%xmm0, %xmm1
	movups	%xmm1, 16(%rsi)
	mulps	%xmm0, %xmm2
	movups	%xmm2, 32(%rsi)
	mulps	%xmm0, %xmm3
	movups	%xmm3, 48(%rsi)
	mulps	%xmm0, %xmm4
	movups	%xmm4, 64(%rsi)
	movups	80(%rsi), %xmm1
	mulps	%xmm0, %xmm1
	movups	%xmm1, 80(%rsi)
	movups	96(%rsi), %xmm1
	mulps	%xmm0, %xmm1
	movups	%xmm1, 96(%rsi)
	movups	112(%rsi), %xmm1
	mulps	%xmm0, %xmm1
	movups	%xmm1, 112(%rsi)
	movups	128(%rsi), %xmm1
	mulps	%xmm0, %xmm1
	movups	%xmm1, 128(%rsi)
	movups	144(%rsi), %xmm1
	mulps	%xmm0, %xmm1
	movups	%xmm1, 144(%rsi)
	movups	160(%rsi), %xmm1
	mulps	%xmm0, %xmm1
	movups	%xmm1, 160(%rsi)
	movups	176(%rsi), %xmm1
	mulps	%xmm0, %xmm1
	movups	%xmm1, 176(%rsi)
	movups	192(%rsi), %xmm1
	mulps	%xmm0, %xmm1
	movups	%xmm1, 192(%rsi)
	movups	208(%rsi), %xmm1
	mulps	%xmm0, %xmm1
	movups	%xmm1, 208(%rsi)
	movups	224(%rsi), %xmm1
	mulps	%xmm0, %xmm1
	movups	%xmm1, 224(%rsi)
	movups	240(%rsi), %xmm1
	mulps	%xmm0, %xmm1
	movups	%xmm1, 240(%rsi)
	movups	256(%rsi), %xmm1
	mulps	%xmm0, %xmm1
	movups	%xmm1, 256(%rsi)
	movups	272(%rsi), %xmm1
	mulps	%xmm0, %xmm1
	movups	%xmm1, 272(%rsi)
	movups	288(%rsi), %xmm1
	mulps	%xmm0, %xmm1
	movups	%xmm1, 288(%rsi)
	movups	304(%rsi), %xmm1
	mulps	%xmm0, %xmm1
	movups	%xmm1, 304(%rsi)
	movups	320(%rsi), %xmm1
	mulps	%xmm0, %xmm1
	movups	%xmm1, 320(%rsi)
	movups	336(%rsi), %xmm1
	mulps	%xmm0, %xmm1
	movups	%xmm1, 336(%rsi)
	movups	352(%rsi), %xmm1
	mulps	%xmm0, %xmm1
	movups	%xmm1, 352(%rsi)
	movups	368(%rsi), %xmm1
	mulps	%xmm0, %xmm1
	movups	%xmm1, 368(%rsi)
	movups	384(%rsi), %xmm1
	mulps	%xmm0, %xmm1
	movups	%xmm1, 384(%rsi)
	movups	400(%rsi), %xmm1
	mulps	%xmm0, %xmm1
	movups	%xmm1, 400(%rsi)
	movups	416(%rsi), %xmm1
	mulps	%xmm0, %xmm1
	movups	%xmm1, 416(%rsi)
	movups	432(%rsi), %xmm1
	mulps	%xmm0, %xmm1
	movups	%xmm1, 432(%rsi)
	movups	448(%rsi), %xmm1
	mulps	%xmm0, %xmm1
	movups	%xmm1, 448(%rsi)
	movups	464(%rsi), %xmm1
	mulps	%xmm0, %xmm1
	movups	%xmm1, 464(%rsi)
	movups	480(%rsi), %xmm1
	mulps	%xmm0, %xmm1
	movups	%xmm1, 480(%rsi)
	movups	496(%rsi), %xmm1
	mulps	%xmm0, %xmm1
	movups	%xmm1, 496(%rsi)
	incq	%rdx
	addq	$512, %rcx                      # imm = 0x200
	cmpq	$128, %rdx
	je	.LBB13_7
.LBB13_3:                               # %.preheader
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB13_5 Depth 2
	movq	%rdx, %rsi
	shlq	$9, %rsi
	addq	%r14, %rsi
	testq	%rdx, %rdx
	je	.LBB13_6
# %bb.4:                                # %.lr.ph.i.preheader
                                        #   in Loop: Header=BB13_3 Depth=1
	movups	496(%rsi), %xmm2
	movups	464(%rsi), %xmm5
	movups	432(%rsi), %xmm11
	movups	400(%rsi), %xmm8
	movups	368(%rsi), %xmm10
	movups	336(%rsi), %xmm1
	movups	304(%rsi), %xmm12
	movups	272(%rsi), %xmm13
	movups	240(%rsi), %xmm0
	movaps	%xmm0, 32(%rsp)                 # 16-byte Spill
	movups	208(%rsi), %xmm0
	movaps	%xmm0, 256(%rsp)                # 16-byte Spill
	movups	176(%rsi), %xmm0
	movaps	%xmm0, 16(%rsp)                 # 16-byte Spill
	movups	144(%rsi), %xmm0
	movaps	%xmm0, 272(%rsp)                # 16-byte Spill
	movups	112(%rsi), %xmm3
	movaps	%xmm3, 240(%rsp)                # 16-byte Spill
	movups	80(%rsi), %xmm3
	movaps	%xmm3, 224(%rsp)                # 16-byte Spill
	movups	(%rsi), %xmm3
	movaps	%xmm3, 384(%rsp)                # 16-byte Spill
	movups	16(%rsi), %xmm3
	movaps	%xmm3, 208(%rsp)                # 16-byte Spill
	movl	$0, %edi
	movups	32(%rsi), %xmm3
	movaps	%xmm3, 368(%rsp)                # 16-byte Spill
	movups	48(%rsi), %xmm3
	movaps	%xmm3, 192(%rsp)                # 16-byte Spill
	movups	480(%rsi), %xmm3
	movaps	%xmm3, 176(%rsp)                # 16-byte Spill
	movups	448(%rsi), %xmm3
	movaps	%xmm3, 160(%rsp)                # 16-byte Spill
	movups	416(%rsi), %xmm3
	movaps	%xmm3, 144(%rsp)                # 16-byte Spill
	movups	384(%rsi), %xmm3
	movaps	%xmm3, 128(%rsp)                # 16-byte Spill
	movups	352(%rsi), %xmm0
	movaps	%xmm0, 112(%rsp)                # 16-byte Spill
	movups	320(%rsi), %xmm0
	movaps	%xmm0, 96(%rsp)                 # 16-byte Spill
	movups	288(%rsi), %xmm0
	movaps	%xmm0, 80(%rsp)                 # 16-byte Spill
	movups	256(%rsi), %xmm0
	movaps	%xmm0, 64(%rsp)                 # 16-byte Spill
	movups	224(%rsi), %xmm0
	movaps	%xmm0, 48(%rsp)                 # 16-byte Spill
	movups	192(%rsi), %xmm0
	movaps	%xmm0, 352(%rsp)                # 16-byte Spill
	movups	160(%rsi), %xmm0
	movaps	%xmm0, 336(%rsp)                # 16-byte Spill
	movups	128(%rsi), %xmm0
	movaps	%xmm0, 320(%rsp)                # 16-byte Spill
	movups	96(%rsi), %xmm0
	movaps	%xmm0, 304(%rsp)                # 16-byte Spill
	movups	64(%rsi), %xmm0
	movaps	%xmm0, 288(%rsp)                # 16-byte Spill
	.p2align	4
.LBB13_5:                               # %.lr.ph.i
                                        #   Parent Loop BB13_3 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movaps	%xmm1, 448(%rsp)                # 16-byte Spill
	movaps	%xmm10, 464(%rsp)               # 16-byte Spill
	movaps	%xmm8, 480(%rsp)                # 16-byte Spill
	movaps	%xmm11, 496(%rsp)               # 16-byte Spill
	movaps	%xmm5, 512(%rsp)                # 16-byte Spill
	movaps	%xmm2, 528(%rsp)                # 16-byte Spill
	movups	(%rax,%rdi), %xmm3
	mulps	%xmm2, %xmm3
	movups	-32(%rax,%rdi), %xmm9
	mulps	%xmm5, %xmm9
	movups	-64(%rax,%rdi), %xmm5
	mulps	%xmm11, %xmm5
	movups	-96(%rax,%rdi), %xmm7
	mulps	%xmm8, %xmm7
	movups	-128(%rax,%rdi), %xmm2
	mulps	%xmm10, %xmm2
	movups	-160(%rax,%rdi), %xmm10
	mulps	%xmm1, %xmm10
	movups	-192(%rax,%rdi), %xmm6
	movaps	%xmm12, 432(%rsp)               # 16-byte Spill
	mulps	%xmm12, %xmm6
	movups	-224(%rax,%rdi), %xmm8
	movaps	%xmm13, 416(%rsp)               # 16-byte Spill
	mulps	%xmm13, %xmm8
	movups	-256(%rax,%rdi), %xmm4
	movaps	32(%rsp), %xmm0                 # 16-byte Reload
	mulps	%xmm0, %xmm4
	movups	-288(%rax,%rdi), %xmm11
	addps	%xmm7, %xmm2
	movaps	256(%rsp), %xmm0                # 16-byte Reload
	mulps	%xmm0, %xmm11
	movups	-320(%rax,%rdi), %xmm7
	movaps	16(%rsp), %xmm0                 # 16-byte Reload
	mulps	%xmm0, %xmm7
	addps	%xmm8, %xmm4
	movups	-352(%rax,%rdi), %xmm8
	movaps	272(%rsp), %xmm0                # 16-byte Reload
	mulps	%xmm0, %xmm8
	addps	%xmm7, %xmm8
	addps	%xmm5, %xmm2
	movups	-384(%rax,%rdi), %xmm5
	mulps	240(%rsp), %xmm5                # 16-byte Folded Reload
	movups	-416(%rax,%rdi), %xmm12
	addps	%xmm6, %xmm4
	mulps	224(%rsp), %xmm12               # 16-byte Folded Reload
	addps	%xmm5, %xmm12
	movups	-480(%rax,%rdi), %xmm5
	movaps	%xmm5, 400(%rsp)                # 16-byte Spill
	addps	%xmm11, %xmm8
	movups	-448(%rax,%rdi), %xmm6
	mulps	192(%rsp), %xmm6                # 16-byte Folded Reload
	mulps	208(%rsp), %xmm5                # 16-byte Folded Reload
	addps	%xmm6, %xmm5
	movups	-496(%rax,%rdi), %xmm1
	addps	%xmm12, %xmm5
	addps	%xmm9, %xmm2
	movups	-112(%rax,%rdi), %xmm11
	mulps	128(%rsp), %xmm11               # 16-byte Folded Reload
	movups	-144(%rax,%rdi), %xmm9
	mulps	112(%rsp), %xmm9                # 16-byte Folded Reload
	addps	%xmm11, %xmm9
	movups	-80(%rax,%rdi), %xmm11
	mulps	144(%rsp), %xmm11               # 16-byte Folded Reload
	addps	%xmm10, %xmm4
	addps	%xmm11, %xmm9
	movups	-240(%rax,%rdi), %xmm11
	movups	-272(%rax,%rdi), %xmm10
	mulps	64(%rsp), %xmm11                # 16-byte Folded Reload
	mulps	48(%rsp), %xmm10                # 16-byte Folded Reload
	addps	%xmm11, %xmm10
	movups	-208(%rax,%rdi), %xmm11
	mulps	80(%rsp), %xmm11                # 16-byte Folded Reload
	addps	%xmm11, %xmm10
	movups	-48(%rax,%rdi), %xmm11
	mulps	160(%rsp), %xmm11               # 16-byte Folded Reload
	addps	%xmm11, %xmm9
	movups	-176(%rax,%rdi), %xmm11
	mulps	96(%rsp), %xmm11                # 16-byte Folded Reload
	addps	%xmm11, %xmm10
	addps	%xmm8, %xmm5
	movups	-336(%rax,%rdi), %xmm11
	movaps	336(%rsp), %xmm14               # 16-byte Reload
	mulps	%xmm14, %xmm11
	movups	-368(%rax,%rdi), %xmm8
	movaps	320(%rsp), %xmm7                # 16-byte Reload
	mulps	%xmm7, %xmm8
	addps	%xmm11, %xmm8
	movups	-16(%rax,%rdi), %xmm11
	mulps	176(%rsp), %xmm11               # 16-byte Folded Reload
	addps	%xmm3, %xmm2
	movups	-304(%rax,%rdi), %xmm3
	movaps	352(%rsp), %xmm15               # 16-byte Reload
	mulps	%xmm15, %xmm3
	addps	%xmm3, %xmm8
	addps	%xmm11, %xmm9
	movups	-400(%rax,%rdi), %xmm3
	movaps	304(%rsp), %xmm6                # 16-byte Reload
	mulps	%xmm6, %xmm3
	movups	-432(%rax,%rdi), %xmm11
	movaps	288(%rsp), %xmm0                # 16-byte Reload
	mulps	%xmm0, %xmm11
	addps	%xmm3, %xmm11
	movups	-464(%rax,%rdi), %xmm3
	movaps	368(%rsp), %xmm13               # 16-byte Reload
	mulps	%xmm13, %xmm3
	addps	%xmm4, %xmm5
	movaps	%xmm1, %xmm4
	movaps	384(%rsp), %xmm12               # 16-byte Reload
	mulps	%xmm12, %xmm4
	addps	%xmm3, %xmm4
	addps	%xmm2, %xmm5
	addps	%xmm11, %xmm4
	addps	%xmm8, %xmm4
	movaps	480(%rsp), %xmm8                # 16-byte Reload
	addps	%xmm10, %xmm4
	movaps	464(%rsp), %xmm10               # 16-byte Reload
	addps	%xmm9, %xmm4
	addps	%xmm5, %xmm4
	movaps	512(%rsp), %xmm5                # 16-byte Reload
	movaps	%xmm4, %xmm3
	unpckhpd	%xmm4, %xmm3                    # xmm3 = xmm3[1],xmm4[1]
	addps	%xmm4, %xmm3
	movaps	%xmm3, %xmm2
	shufps	$85, %xmm3, %xmm2               # xmm2 = xmm2[1,1],xmm3[1,1]
	addss	%xmm3, %xmm2
	shufps	$0, %xmm2, %xmm2                # xmm2 = xmm2[0,0,0,0]
	mulps	%xmm2, %xmm1
	movaps	400(%rsp), %xmm3                # 16-byte Reload
	mulps	%xmm2, %xmm3
	subps	%xmm1, %xmm12
	movaps	496(%rsp), %xmm11               # 16-byte Reload
	movaps	208(%rsp), %xmm4                # 16-byte Reload
	subps	%xmm3, %xmm4
	movaps	%xmm12, 384(%rsp)               # 16-byte Spill
	movups	%xmm12, (%rsi)
	movaps	%xmm4, 208(%rsp)                # 16-byte Spill
	movups	%xmm4, 16(%rsi)
	movups	-464(%rax,%rdi), %xmm3
	movups	-448(%rax,%rdi), %xmm4
	mulps	%xmm2, %xmm3
	mulps	%xmm2, %xmm4
	subps	%xmm3, %xmm13
	movaps	192(%rsp), %xmm3                # 16-byte Reload
	subps	%xmm4, %xmm3
	movaps	%xmm13, 368(%rsp)               # 16-byte Spill
	movups	%xmm13, 32(%rsi)
	movaps	%xmm3, 192(%rsp)                # 16-byte Spill
	movups	%xmm3, 48(%rsi)
	movups	-432(%rax,%rdi), %xmm3
	movups	-416(%rax,%rdi), %xmm4
	mulps	%xmm2, %xmm3
	mulps	%xmm2, %xmm4
	subps	%xmm3, %xmm0
	movaps	224(%rsp), %xmm3                # 16-byte Reload
	subps	%xmm4, %xmm3
	movaps	%xmm0, 288(%rsp)                # 16-byte Spill
	movups	%xmm0, 64(%rsi)
	movaps	%xmm3, 224(%rsp)                # 16-byte Spill
	movups	%xmm3, 80(%rsi)
	movups	-400(%rax,%rdi), %xmm3
	movups	-384(%rax,%rdi), %xmm4
	mulps	%xmm2, %xmm3
	mulps	%xmm2, %xmm4
	subps	%xmm3, %xmm6
	movaps	240(%rsp), %xmm0                # 16-byte Reload
	subps	%xmm4, %xmm0
	movaps	%xmm6, 304(%rsp)                # 16-byte Spill
	movups	%xmm6, 96(%rsi)
	movaps	%xmm0, 240(%rsp)                # 16-byte Spill
	movups	%xmm0, 112(%rsi)
	movups	-368(%rax,%rdi), %xmm3
	movups	-352(%rax,%rdi), %xmm4
	mulps	%xmm2, %xmm3
	mulps	%xmm2, %xmm4
	subps	%xmm3, %xmm7
	movaps	272(%rsp), %xmm0                # 16-byte Reload
	subps	%xmm4, %xmm0
	movaps	%xmm7, 320(%rsp)                # 16-byte Spill
	movups	%xmm7, 128(%rsi)
	movaps	%xmm0, 272(%rsp)                # 16-byte Spill
	movups	%xmm0, 144(%rsi)
	movups	-336(%rax,%rdi), %xmm3
	movups	-320(%rax,%rdi), %xmm4
	mulps	%xmm2, %xmm3
	mulps	%xmm2, %xmm4
	subps	%xmm3, %xmm14
	movaps	16(%rsp), %xmm0                 # 16-byte Reload
	subps	%xmm4, %xmm0
	movaps	%xmm14, 336(%rsp)               # 16-byte Spill
	movups	%xmm14, 160(%rsi)
	movaps	%xmm0, 16(%rsp)                 # 16-byte Spill
	movups	%xmm0, 176(%rsi)
	movups	-304(%rax,%rdi), %xmm3
	movups	-288(%rax,%rdi), %xmm4
	mulps	%xmm2, %xmm3
	mulps	%xmm2, %xmm4
	subps	%xmm3, %xmm15
	movaps	256(%rsp), %xmm0                # 16-byte Reload
	subps	%xmm4, %xmm0
	movaps	%xmm15, 352(%rsp)               # 16-byte Spill
	movups	%xmm15, 192(%rsi)
	movaps	%xmm0, 256(%rsp)                # 16-byte Spill
	movups	%xmm0, 208(%rsi)
	movups	-272(%rax,%rdi), %xmm3
	movups	-256(%rax,%rdi), %xmm4
	mulps	%xmm2, %xmm3
	mulps	%xmm2, %xmm4
	movaps	48(%rsp), %xmm1                 # 16-byte Reload
	subps	%xmm3, %xmm1
	movaps	32(%rsp), %xmm0                 # 16-byte Reload
	subps	%xmm4, %xmm0
	movaps	%xmm1, 48(%rsp)                 # 16-byte Spill
	movups	%xmm1, 224(%rsi)
	movaps	%xmm0, 32(%rsp)                 # 16-byte Spill
	movups	%xmm0, 240(%rsi)
	movups	-240(%rax,%rdi), %xmm3
	movups	-224(%rax,%rdi), %xmm4
	mulps	%xmm2, %xmm3
	mulps	%xmm2, %xmm4
	movaps	64(%rsp), %xmm1                 # 16-byte Reload
	subps	%xmm3, %xmm1
	movaps	416(%rsp), %xmm0                # 16-byte Reload
	subps	%xmm4, %xmm0
	movaps	%xmm1, 64(%rsp)                 # 16-byte Spill
	movups	%xmm1, 256(%rsi)
	movaps	%xmm0, %xmm13
	movups	%xmm0, 272(%rsi)
	movups	-208(%rax,%rdi), %xmm3
	movups	-192(%rax,%rdi), %xmm4
	mulps	%xmm2, %xmm3
	mulps	%xmm2, %xmm4
	movaps	80(%rsp), %xmm1                 # 16-byte Reload
	subps	%xmm3, %xmm1
	movaps	432(%rsp), %xmm0                # 16-byte Reload
	subps	%xmm4, %xmm0
	movaps	%xmm1, 80(%rsp)                 # 16-byte Spill
	movups	%xmm1, 288(%rsi)
	movaps	%xmm0, %xmm12
	movups	%xmm0, 304(%rsi)
	movups	-176(%rax,%rdi), %xmm3
	movups	-160(%rax,%rdi), %xmm4
	mulps	%xmm2, %xmm3
	mulps	%xmm2, %xmm4
	movaps	96(%rsp), %xmm0                 # 16-byte Reload
	subps	%xmm3, %xmm0
	movaps	448(%rsp), %xmm3                # 16-byte Reload
	subps	%xmm4, %xmm3
	movaps	%xmm0, 96(%rsp)                 # 16-byte Spill
	movups	%xmm0, 320(%rsi)
	movaps	%xmm3, %xmm1
	movups	%xmm3, 336(%rsi)
	movups	-144(%rax,%rdi), %xmm3
	movups	-128(%rax,%rdi), %xmm4
	mulps	%xmm2, %xmm3
	mulps	%xmm2, %xmm4
	movaps	112(%rsp), %xmm0                # 16-byte Reload
	subps	%xmm3, %xmm0
	subps	%xmm4, %xmm10
	movaps	%xmm0, 112(%rsp)                # 16-byte Spill
	movups	%xmm0, 352(%rsi)
	movups	%xmm10, 368(%rsi)
	movups	-112(%rax,%rdi), %xmm3
	movups	-96(%rax,%rdi), %xmm4
	mulps	%xmm2, %xmm3
	mulps	%xmm2, %xmm4
	movaps	128(%rsp), %xmm0                # 16-byte Reload
	subps	%xmm3, %xmm0
	subps	%xmm4, %xmm8
	movaps	%xmm0, 128(%rsp)                # 16-byte Spill
	movups	%xmm0, 384(%rsi)
	movups	%xmm8, 400(%rsi)
	movups	-80(%rax,%rdi), %xmm3
	movups	-64(%rax,%rdi), %xmm4
	mulps	%xmm2, %xmm3
	mulps	%xmm2, %xmm4
	movaps	144(%rsp), %xmm0                # 16-byte Reload
	subps	%xmm3, %xmm0
	subps	%xmm4, %xmm11
	movaps	%xmm0, 144(%rsp)                # 16-byte Spill
	movups	%xmm0, 416(%rsi)
	movups	%xmm11, 432(%rsi)
	movups	-48(%rax,%rdi), %xmm3
	movups	-32(%rax,%rdi), %xmm4
	mulps	%xmm2, %xmm3
	mulps	%xmm2, %xmm4
	movaps	160(%rsp), %xmm0                # 16-byte Reload
	subps	%xmm3, %xmm0
	subps	%xmm4, %xmm5
	movaps	%xmm0, 160(%rsp)                # 16-byte Spill
	movups	%xmm0, 448(%rsi)
	movups	%xmm5, 464(%rsi)
	movups	-16(%rax,%rdi), %xmm3
	movups	(%rax,%rdi), %xmm4
	mulps	%xmm2, %xmm3
	mulps	%xmm2, %xmm4
	movaps	528(%rsp), %xmm2                # 16-byte Reload
	movaps	176(%rsp), %xmm0                # 16-byte Reload
	subps	%xmm3, %xmm0
	subps	%xmm4, %xmm2
	movaps	%xmm0, 176(%rsp)                # 16-byte Spill
	movups	%xmm0, 480(%rsi)
	movups	%xmm2, 496(%rsi)
	addq	$512, %rdi                      # imm = 0x200
	cmpq	%rdi, %rcx
	jne	.LBB13_5
	jmp	.LBB13_6
.LBB13_7:                               # %_ZL20gram_schmidt_inplacePfi.exit
	movq	$0, 8(%rsp)
	leaq	8(%rsp), %rdi
	movl	$65536, %esi                    # imm = 0x10000
	callq	hipMalloc@PLT
	movq	8(%rsp), %rdi
	movl	$65536, %edx                    # imm = 0x10000
	movq	%r14, %rsi
	movl	$1, %ecx
	callq	hipMemcpy@PLT
	movq	%r14, %rdi
	callq	free@PLT
	movq	8(%rsp), %rax
	movq	%rax, (%rbx)
	addq	$552, %rsp                      # imm = 0x228
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end13:
	.size	tqm_alloc_rotation, .Lfunc_end13-tqm_alloc_rotation
	.cfi_endproc
                                        # -- End function
	.section	.rodata.cst8,"aM",@progbits,8
	.p2align	3, 0x0                          # -- Begin function tqm_alloc_qjl_matrix
.LCPI14_0:
	.quad	0x3ca0000000000000              # double 1.1102230246251565E-16
.LCPI14_1:
	.quad	0x39b4484bfeebc2a0              # double 1.0000000000000001E-30
.LCPI14_2:
	.quad	0xc000000000000000              # double -2
.LCPI14_3:
	.quad	0x3cc921fb54442d18              # double 6.9757369960172635E-16
	.section	.rodata.cst4,"aM",@progbits,4
	.p2align	2, 0x0
.LCPI14_4:
	.long	0x3db504f3                      # float 0.0883883461
	.text
	.globl	tqm_alloc_qjl_matrix
	.p2align	4
	.type	tqm_alloc_qjl_matrix,@function
tqm_alloc_qjl_matrix:                   # @tqm_alloc_qjl_matrix
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	subq	$24, %rsp
	.cfi_def_cfa_offset 80
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movq	%rsi, %rbx
	movq	%rdi, %r15
	movabsq	$6364136223846793005, %r12      # imm = 0x5851F42D4C957F2D
	movabsq	$1442695040888963407, %r13      # imm = 0x14057B7EF767814F
	movl	$65536, %edi                    # imm = 0x10000
	callq	malloc@PLT
	movq	%rax, %r14
	movabsq	$-4688283849052983283, %rax     # imm = 0xBEEFDEADCAFEF00D
	xorq	%r15, %rax
	xorl	%r15d, %r15d
	.p2align	4
.LBB14_1:                               # =>This Inner Loop Header: Depth=1
	imulq	%r12, %rax
	addq	%r13, %rax
	movq	%rax, %rbp
	imulq	%r12, %rbp
	addq	%r13, %rbp
	shrq	$11, %rax
	xorps	%xmm0, %xmm0
	cvtsi2sd	%rax, %xmm0
	mulsd	.LCPI14_0(%rip), %xmm0
	maxsd	.LCPI14_1(%rip), %xmm0
	movq	%rbp, %rax
	shrq	$11, %rax
	xorps	%xmm1, %xmm1
	cvtsi2sd	%rax, %xmm1
	movsd	%xmm1, 16(%rsp)                 # 8-byte Spill
	callq	log@PLT
	mulsd	.LCPI14_2(%rip), %xmm0
	sqrtsd	%xmm0, %xmm0
	movsd	%xmm0, 8(%rsp)                  # 8-byte Spill
	movsd	16(%rsp), %xmm0                 # 8-byte Reload
                                        # xmm0 = mem[0],zero
	mulsd	.LCPI14_3(%rip), %xmm0
	callq	cos@PLT
	mulsd	8(%rsp), %xmm0                  # 8-byte Folded Reload
	cvtsd2ss	%xmm0, %xmm0
	mulss	.LCPI14_4(%rip), %xmm0
	movss	%xmm0, (%r14,%r15,4)
	incq	%r15
	movq	%rbp, %rax
	cmpq	$16384, %r15                    # imm = 0x4000
	jne	.LBB14_1
# %bb.2:
	movq	$0, (%rsp)
	movq	%rsp, %rdi
	movl	$65536, %esi                    # imm = 0x10000
	callq	hipMalloc@PLT
	movq	(%rsp), %rdi
	movl	$65536, %edx                    # imm = 0x10000
	movq	%r14, %rsi
	movl	$1, %ecx
	callq	hipMemcpy@PLT
	movq	%r14, %rdi
	callq	free@PLT
	movq	(%rsp), %rax
	movq	%rax, (%rbx)
	addq	$24, %rsp
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end14:
	.size	tqm_alloc_qjl_matrix, .Lfunc_end14-tqm_alloc_qjl_matrix
	.cfi_endproc
                                        # -- End function
	.p2align	4                               # -- Begin function __hip_module_ctor
	.type	__hip_module_ctor,@function
__hip_module_ctor:                      # @__hip_module_ctor
	.cfi_startproc
# %bb.0:
	pushq	%rbx
	.cfi_def_cfa_offset 16
	subq	$32, %rsp
	.cfi_def_cfa_offset 48
	.cfi_offset %rbx, -16
	movq	__hip_gpubin_handle_81ebb66926e4b9a5(%rip), %rbx
	testq	%rbx, %rbx
	jne	.LBB15_2
# %bb.1:
	leaq	__hip_fatbin_wrapper(%rip), %rdi
	callq	__hipRegisterFatBinary@PLT
	movq	%rax, %rbx
	movq	%rax, __hip_gpubin_handle_81ebb66926e4b9a5(%rip)
.LBB15_2:
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rsp)
	movups	%xmm0, (%rsp)
	movq	_Z23tqm_quantize_kernel_tq3PKfS0_P16block_tq3_mi300xi@GOTPCREL(%rip), %rsi
	leaq	.L__unnamed_1(%rip), %rcx
	movq	%rbx, %rdi
	movq	%rcx, %rdx
	movl	$-1, %r8d
	xorl	%r9d, %r9d
	callq	__hipRegisterFunction@PLT
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rsp)
	movups	%xmm0, (%rsp)
	movq	_Z25tqm_dequantize_kernel_tq3PK16block_tq3_mi300xPKfPfi@GOTPCREL(%rip), %rsi
	leaq	.L__unnamed_2(%rip), %rcx
	movq	%rbx, %rdi
	movq	%rcx, %rdx
	movl	$-1, %r8d
	xorl	%r9d, %r9d
	callq	__hipRegisterFunction@PLT
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rsp)
	movups	%xmm0, (%rsp)
	movq	_Z24tqm_fused_dot_kernel_tq3PKfPK16block_tq3_mi300xPfii@GOTPCREL(%rip), %rsi
	leaq	.L__unnamed_3(%rip), %rcx
	movq	%rbx, %rdi
	movq	%rcx, %rdx
	movl	$-1, %r8d
	xorl	%r9d, %r9d
	callq	__hipRegisterFunction@PLT
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rsp)
	movups	%xmm0, (%rsp)
	movq	_Z23tqm_quantize_kernel_tq4PKfS0_P16block_tq4_mi300xi@GOTPCREL(%rip), %rsi
	leaq	.L__unnamed_4(%rip), %rcx
	movq	%rbx, %rdi
	movq	%rcx, %rdx
	movl	$-1, %r8d
	xorl	%r9d, %r9d
	callq	__hipRegisterFunction@PLT
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rsp)
	movups	%xmm0, (%rsp)
	movq	_Z25tqm_dequantize_kernel_tq4PK16block_tq4_mi300xPKfPfi@GOTPCREL(%rip), %rsi
	leaq	.L__unnamed_5(%rip), %rcx
	movq	%rbx, %rdi
	movq	%rcx, %rdx
	movl	$-1, %r8d
	xorl	%r9d, %r9d
	callq	__hipRegisterFunction@PLT
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rsp)
	movups	%xmm0, (%rsp)
	movq	_Z14tqm_qjl_kernelPKfPK16block_tq3_mi300xS0_S0_P16block_qjl_mi300xi@GOTPCREL(%rip), %rsi
	leaq	.L__unnamed_6(%rip), %rcx
	movq	%rbx, %rdi
	movq	%rcx, %rdx
	movl	$-1, %r8d
	xorl	%r9d, %r9d
	callq	__hipRegisterFunction@PLT
	movl	$0, 8(%rsp)
	movl	$1, (%rsp)
	leaq	d_cb3(%rip), %rsi
	leaq	.L__unnamed_7(%rip), %rcx
	movl	$32, %r9d
	movq	%rbx, %rdi
	movq	%rcx, %rdx
	xorl	%r8d, %r8d
	callq	__hipRegisterVar@PLT
	movl	$0, 8(%rsp)
	movl	$1, (%rsp)
	leaq	d_cb4(%rip), %rsi
	leaq	.L__unnamed_8(%rip), %rcx
	movl	$64, %r9d
	movq	%rbx, %rdi
	movq	%rcx, %rdx
	xorl	%r8d, %r8d
	callq	__hipRegisterVar@PLT
	movl	$0, 8(%rsp)
	movl	$1, (%rsp)
	leaq	d_cb2(%rip), %rsi
	leaq	.L__unnamed_9(%rip), %rcx
	movl	$16, %r9d
	movq	%rbx, %rdi
	movq	%rcx, %rdx
	xorl	%r8d, %r8d
	callq	__hipRegisterVar@PLT
	leaq	__hip_module_dtor(%rip), %rdi
	addq	$32, %rsp
	.cfi_def_cfa_offset 16
	popq	%rbx
	.cfi_def_cfa_offset 8
	jmp	atexit@PLT                      # TAILCALL
.Lfunc_end15:
	.size	__hip_module_ctor, .Lfunc_end15-__hip_module_ctor
	.cfi_endproc
                                        # -- End function
	.p2align	4                               # -- Begin function __hip_module_dtor
	.type	__hip_module_dtor,@function
__hip_module_dtor:                      # @__hip_module_dtor
	.cfi_startproc
# %bb.0:
	movq	__hip_gpubin_handle_81ebb66926e4b9a5(%rip), %rdi
	testq	%rdi, %rdi
	je	.LBB16_2
# %bb.1:
	pushq	%rax
	.cfi_def_cfa_offset 16
	callq	__hipUnregisterFatBinary@PLT
	movq	$0, __hip_gpubin_handle_81ebb66926e4b9a5(%rip)
	addq	$8, %rsp
	.cfi_def_cfa_offset 8
.LBB16_2:
	retq
.Lfunc_end16:
	.size	__hip_module_dtor, .Lfunc_end16-__hip_module_dtor
	.cfi_endproc
                                        # -- End function
	.type	d_cb3,@object                   # @d_cb3
	.local	d_cb3
	.comm	d_cb3,32,16
	.type	d_cb4,@object                   # @d_cb4
	.local	d_cb4
	.comm	d_cb4,64,16
	.type	d_cb2,@object                   # @d_cb2
	.local	d_cb2
	.comm	d_cb2,16,16
	.type	_Z23tqm_quantize_kernel_tq3PKfS0_P16block_tq3_mi300xi,@object # @_Z23tqm_quantize_kernel_tq3PKfS0_P16block_tq3_mi300xi
	.section	.data.rel.ro,"aw",@progbits
	.globl	_Z23tqm_quantize_kernel_tq3PKfS0_P16block_tq3_mi300xi
	.p2align	3, 0x0
_Z23tqm_quantize_kernel_tq3PKfS0_P16block_tq3_mi300xi:
	.quad	_Z38__device_stub__tqm_quantize_kernel_tq3PKfS0_P16block_tq3_mi300xi
	.size	_Z23tqm_quantize_kernel_tq3PKfS0_P16block_tq3_mi300xi, 8

	.type	_Z25tqm_dequantize_kernel_tq3PK16block_tq3_mi300xPKfPfi,@object # @_Z25tqm_dequantize_kernel_tq3PK16block_tq3_mi300xPKfPfi
	.globl	_Z25tqm_dequantize_kernel_tq3PK16block_tq3_mi300xPKfPfi
	.p2align	3, 0x0
_Z25tqm_dequantize_kernel_tq3PK16block_tq3_mi300xPKfPfi:
	.quad	_Z40__device_stub__tqm_dequantize_kernel_tq3PK16block_tq3_mi300xPKfPfi
	.size	_Z25tqm_dequantize_kernel_tq3PK16block_tq3_mi300xPKfPfi, 8

	.type	_Z24tqm_fused_dot_kernel_tq3PKfPK16block_tq3_mi300xPfii,@object # @_Z24tqm_fused_dot_kernel_tq3PKfPK16block_tq3_mi300xPfii
	.globl	_Z24tqm_fused_dot_kernel_tq3PKfPK16block_tq3_mi300xPfii
	.p2align	3, 0x0
_Z24tqm_fused_dot_kernel_tq3PKfPK16block_tq3_mi300xPfii:
	.quad	_Z39__device_stub__tqm_fused_dot_kernel_tq3PKfPK16block_tq3_mi300xPfii
	.size	_Z24tqm_fused_dot_kernel_tq3PKfPK16block_tq3_mi300xPfii, 8

	.type	_Z23tqm_quantize_kernel_tq4PKfS0_P16block_tq4_mi300xi,@object # @_Z23tqm_quantize_kernel_tq4PKfS0_P16block_tq4_mi300xi
	.globl	_Z23tqm_quantize_kernel_tq4PKfS0_P16block_tq4_mi300xi
	.p2align	3, 0x0
_Z23tqm_quantize_kernel_tq4PKfS0_P16block_tq4_mi300xi:
	.quad	_Z38__device_stub__tqm_quantize_kernel_tq4PKfS0_P16block_tq4_mi300xi
	.size	_Z23tqm_quantize_kernel_tq4PKfS0_P16block_tq4_mi300xi, 8

	.type	_Z25tqm_dequantize_kernel_tq4PK16block_tq4_mi300xPKfPfi,@object # @_Z25tqm_dequantize_kernel_tq4PK16block_tq4_mi300xPKfPfi
	.globl	_Z25tqm_dequantize_kernel_tq4PK16block_tq4_mi300xPKfPfi
	.p2align	3, 0x0
_Z25tqm_dequantize_kernel_tq4PK16block_tq4_mi300xPKfPfi:
	.quad	_Z40__device_stub__tqm_dequantize_kernel_tq4PK16block_tq4_mi300xPKfPfi
	.size	_Z25tqm_dequantize_kernel_tq4PK16block_tq4_mi300xPKfPfi, 8

	.type	_Z14tqm_qjl_kernelPKfPK16block_tq3_mi300xS0_S0_P16block_qjl_mi300xi,@object # @_Z14tqm_qjl_kernelPKfPK16block_tq3_mi300xS0_S0_P16block_qjl_mi300xi
	.globl	_Z14tqm_qjl_kernelPKfPK16block_tq3_mi300xS0_S0_P16block_qjl_mi300xi
	.p2align	3, 0x0
_Z14tqm_qjl_kernelPKfPK16block_tq3_mi300xS0_S0_P16block_qjl_mi300xi:
	.quad	_Z29__device_stub__tqm_qjl_kernelPKfPK16block_tq3_mi300xS0_S0_P16block_qjl_mi300xi
	.size	_Z14tqm_qjl_kernelPKfPK16block_tq3_mi300xS0_S0_P16block_qjl_mi300xi, 8

	.type	_ZL19TQ3_CODEBOOK_MI300X,@object # @_ZL19TQ3_CODEBOOK_MI300X
	.section	.rodata,"a",@progbits
	.p2align	4, 0x0
_ZL19TQ3_CODEBOOK_MI300X:
	.long	0xbe4193cd                      # float -0.189040378
	.long	0xbdf34acd                      # float -0.118795015
	.long	0xbd89469e                      # float -0.0670292228
	.long	0xbcb22c74                      # float -0.0217497125
	.long	0x3cb22c74                      # float 0.0217497125
	.long	0x3d89469e                      # float 0.0670292228
	.long	0x3df34acd                      # float 0.118795015
	.long	0x3e4193cd                      # float 0.189040378
	.size	_ZL19TQ3_CODEBOOK_MI300X, 32

	.type	_ZL19TQ4_CODEBOOK_MI300X,@object # @_ZL19TQ4_CODEBOOK_MI300X
	.p2align	4, 0x0
_ZL19TQ4_CODEBOOK_MI300X:
	.long	0xbe755cfd                      # float -0.239612535
	.long	0xbe3b9133                      # float -0.183171079
	.long	0xbe13c5ec                      # float -0.1443097
	.long	0xbde6f1ca                      # float -0.112765864
	.long	0xbdae3bb4                      # float -0.0850748121
	.long	0xbd743579                      # float -0.059621308
	.long	0xbd10f54a                      # float -0.0353901759
	.long	0xbc403b24                      # float -0.0117328502
	.long	0x3c403b24                      # float 0.0117328502
	.long	0x3d10f54a                      # float 0.0353901759
	.long	0x3d743579                      # float 0.059621308
	.long	0x3dae3bb4                      # float 0.0850748121
	.long	0x3de6f1ca                      # float 0.112765864
	.long	0x3e13c5ec                      # float 0.1443097
	.long	0x3e3b9133                      # float 0.183171079
	.long	0x3e755cfd                      # float 0.239612535
	.size	_ZL19TQ4_CODEBOOK_MI300X, 64

	.type	_ZL19TQ2_CODEBOOK_MI300X,@object # @_ZL19TQ2_CODEBOOK_MI300X
	.p2align	4, 0x0
_ZL19TQ2_CODEBOOK_MI300X:
	.long	0xbe084f2c                      # float -0.133114517
	.long	0xbd23f3d7                      # float -0.0400274657
	.long	0x3d23f3d7                      # float 0.0400274657
	.long	0x3e084f2c                      # float 0.133114517
	.size	_ZL19TQ2_CODEBOOK_MI300X, 16

	.type	.L__unnamed_1,@object           # @0
	.section	.rodata.str1.1,"aMS",@progbits,1
.L__unnamed_1:
	.asciz	"_Z23tqm_quantize_kernel_tq3PKfS0_P16block_tq3_mi300xi"
	.size	.L__unnamed_1, 54

	.type	.L__unnamed_2,@object           # @1
.L__unnamed_2:
	.asciz	"_Z25tqm_dequantize_kernel_tq3PK16block_tq3_mi300xPKfPfi"
	.size	.L__unnamed_2, 56

	.type	.L__unnamed_3,@object           # @2
.L__unnamed_3:
	.asciz	"_Z24tqm_fused_dot_kernel_tq3PKfPK16block_tq3_mi300xPfii"
	.size	.L__unnamed_3, 56

	.type	.L__unnamed_4,@object           # @3
.L__unnamed_4:
	.asciz	"_Z23tqm_quantize_kernel_tq4PKfS0_P16block_tq4_mi300xi"
	.size	.L__unnamed_4, 54

	.type	.L__unnamed_5,@object           # @4
.L__unnamed_5:
	.asciz	"_Z25tqm_dequantize_kernel_tq4PK16block_tq4_mi300xPKfPfi"
	.size	.L__unnamed_5, 56

	.type	.L__unnamed_6,@object           # @5
.L__unnamed_6:
	.asciz	"_Z14tqm_qjl_kernelPKfPK16block_tq3_mi300xS0_S0_P16block_qjl_mi300xi"
	.size	.L__unnamed_6, 68

	.type	.L__unnamed_7,@object           # @6
.L__unnamed_7:
	.asciz	"d_cb3"
	.size	.L__unnamed_7, 6

	.type	.L__unnamed_8,@object           # @7
.L__unnamed_8:
	.asciz	"d_cb4"
	.size	.L__unnamed_8, 6

	.type	.L__unnamed_9,@object           # @8
.L__unnamed_9:
	.asciz	"d_cb2"
	.size	.L__unnamed_9, 6

	.type	.L__unnamed_10,@object          # @9
	.section	.hip_fatbin,"a",@progbits
	.p2align	12, 0x0
.L__unnamed_10:
	.asciz	"__CLANG_OFFLOAD_BUNDLE__\002\000\000\000\000\000\000\000\000\020\000\000\000\000\000\000\000\000\000\000\000\000\000\000\036\000\000\000\000\000\000\000host-x86_64-unknown-linux-gnu-\000\020\000\000\000\000\000\0008\225\000\000\000\000\000\000/\000\000\000\000\000\000\000hipv4-amdgcn-amd-amdhsa--gfx942:sramecc+:xnack-\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\177ELF\002\001\001@\004\000\000\000\000\000\000\000\003\000\340\000\001\000\000\000\000\000\000\000\000\000\000\000@\000\000\000\000\000\000\000\370\220\000\000\000\000\000\000L\016\000\000@\0008\000\t\000@\000\021\000\017\000\006\000\000\000\004\000\000\000@\000\000\000\000\000\000\000@\000\000\000\000\000\000\000@\000\000\000\000\000\000\000\370\001\000\000\000\000\000\000\370\001\000\000\000\000\000\000\b\000\000\000\000\000\000\000\001\000\000\000\004\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\0000\035\000\000\000\000\000\0000\035\000\000\000\000\000\000\000\020\000\000\000\000\000\000\001\000\000\000\005\000\000\000\000\036\000\000\000\000\000\000\000.\000\000\000\000\000\000\000.\000\000\000\000\000\000\000U\000\000\000\000\000\000\000U\000\000\000\000\000\000\000\020\000\000\000\000\000\000\001\000\000\000\006\000\000\000\000s\000\000\000\000\000\000\000\223\000\000\000\000\000\000\000\223\000\000\000\000\000\000p\000\000\000\000\000\000\000\000\r\000\000\000\000\000\000\000\020\000\000\000\000\000\000\001\000\000\000\006\000\000\000ps\000\000\000\000\000\000p\243\000\000\000\000\000\000p\243\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\020\000\000\000\000\000\000\002\000\000\000\006\000\000\000\000s\000\000\000\000\000\000\000\223\000\000\000\000\000\000\000\223\000\000\000\000\000\000p\000\000\000\000\000\000\000p\000\000\000\000\000\000\000\b\000\000\000\000\000\000\000R\345td\004\000\000\000\000s\000\000\000\000\000\000\000\223\000\000\000\000\000\000\000\223\000\000\000\000\000\000p\000\000\000\000\000\000\000\000\r\000\000\000\000\000\000\001\000\000\000\000\000\000\000Q\345td\006\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\004\000\000\000\004\000\000\0008\002\000\000\000\000\000\0008\002\000\000\000\000\000\0008\002\000\000\000\000\000\000L\023\000\000\000\000\000\000L\023\000\000\000\000\000\000\004\000\000\000\000\000\000\000\007\000\000\0008\023\000\000 \000\000\000AMDGPU\000\000\203\256amdhsa.kernels\226\336\000\022\253.agpr_count\000\245.args\224\205\256.actual_access\251read_only\256.address_space\246global\247.offset\000\245.size\b\253.value_kind\255global_buffer\205\256.actual_access\251read_only\256.address_space\246global\247.offset\b\245.size\b\253.value_kind\255global_buffer\205\256.actual_access\252write_only\256.address_space\246global\247.offset\020\245.size\b\253.value_kind\255global_buffer\203\247.offset\030\245.size\004\253.value_kind\250by_value\271.group_segment_fixed_size\315\002\f\266.kernarg_segment_align\b\265.kernarg_segment_size\034\251.language\250OpenCL C\261.language_version\222\002\000\270.max_flat_workgroup_size\315\004\000\245.name\3315_Z23tqm_quantize_kernel_tq3PKfS0_P16block_tq3_mi300xi\273.private_segment_fixed_size\315\002\204\253.sgpr_count\026\261.sgpr_spill_count\000\247.symbol\3318_Z23tqm_quantize_kernel_tq3PKfS0_P16block_tq3_mi300xi.kd\270.uniform_work_group_size\001\263.uses_dynamic_stack\302\253.vgpr_count\314\200\261.vgpr_spill_count\314\240\257.wavefront_size@\336\000\022\253.agpr_count\000\245.args\224\205\256.actual_access\251read_only\256.address_space\246global\247.offset\000\245.size\b\253.value_kind\255global_buffer\205\256.actual_access\251read_only\256.address_space\246global\247.offset\b\245.size\b\253.value_kind\255global_buffer\205\256.actual_access\252write_only\256.address_space\246global\247.offset\020\245.size\b\253.value_kind\255global_buffer\203\247.offset\030\245.size\004\253.value_kind\250by_value\271.group_segment_fixed_size\315\002\004\266.kernarg_segment_align\b\265.kernarg_segment_size\034\251.language\250OpenCL C\261.language_version\222\002\000\270.max_flat_workgroup_size\315\004\000\245.name\3317_Z25tqm_dequantize_kernel_tq3PK16block_tq3_mi300xPKfPfi\273.private_segment_fixed_size\000\253.sgpr_count\020\261.sgpr_spill_count\000\247.symbol\331:_Z25tqm_dequantize_kernel_tq3PK16block_tq3_mi300xPKfPfi.kd\270.uniform_work_group_size\001\263.uses_dynamic_stack\302\253.vgpr_countK\261.vgpr_spill_count\000\257.wavefront_size@\336\000\022\253.agpr_count\000\245.args\225\205\256.actual_access\251read_only\256.address_space\246global\247.offset\000\245.size\b\253.value_kind\255global_buffer\205\256.actual_access\251read_only\256.address_space\246global\247.offset\b\245.size\b\253.value_kind\255global_buffer\205\256.actual_access\252write_only\256.address_space\246global\247.offset\020\245.size\b\253.value_kind\255global_buffer\203\247.offset\030\245.size\004\253.value_kind\250by_value\203\247.offset\034\245.size\004\253.value_kind\250by_value\271.group_segment_fixed_size\f\266.kernarg_segment_align\b\265.kernarg_segment_size \251.language\250OpenCL C\261.language_version\222\002\000\270.max_flat_workgroup_size\315\004\000\245.name\3317_Z24tqm_fused_dot_kernel_tq3PKfPK16block_tq3_mi300xPfii\273.private_segment_fixed_size\000\253.sgpr_count\024\261.sgpr_spill_count\000\247.symbol\331:_Z24tqm_fused_dot_kernel_tq3PKfPK16block_tq3_mi300xPfii.kd\270.uniform_work_group_size\001\263.uses_dynamic_stack\302\253.vgpr_count\f\261.vgpr_spill_count\000\257.wavefront_size@\336\000\022\253.agpr_count\000\245.args\224\205\256.actual_access\251read_only\256.address_space\246global\247.offset\000\245.size\b\253.value_kind\255global_buffer\205\256.actual_access\251read_only\256.address_space\246global\247.offset\b\245.size\b\253.value_kind\255global_buffer\205\256.actual_access\252write_only\256.address_space\246global\247.offset\020\245.size\b\253.value_kind\255global_buffer\203\247.offset\030\245.size\004\253.value_kind\250by_value\271.group_segment_fixed_size\315\002\f\266.kernarg_segment_align\b\265.kernarg_segment_size\034\251.language\250OpenCL C\261.language_version\222\002\000\270.max_flat_workgroup_size\315\004\000\245.name\3315_Z23tqm_quantize_kernel_tq4PKfS0_P16block_tq4_mi300xi\273.private_segment_fixed_size\315\002\204\253.sgpr_count\036\261.sgpr_spill_count\000\247.symbol\3318_Z23tqm_quantize_kernel_tq4PKfS0_P16block_tq4_mi300xi.kd\270.uniform_work_group_size\001\263.uses_dynamic_stack\302\253.vgpr_count\314\200\261.vgpr_spill_count\314\240\257.wavefront_size@\336\000\022\253.agpr_count\000\245.args\224\205\256.actual_access\251read_only\256.address_space\246global\247.offset\000\245.size\b\253.value_kind\255global_buffer\205\256.actual_access\251read_only\256.address_space\246global\247.offset\b\245.size\b\253.value_kind\255global_buffer\205\256.actual_access\252write_only\256.address_space\246global\247.offset\020\245.size\b\253.value_kind\255global_buffer\203\247.offset\030\245.size\004\253.value_kind\250by_value\271.group_segment_fixed_size\315\002\004\266.kernarg_segment_align\b\265.kernarg_segment_size\034\251.language\250OpenCL C\261.language_version\222\002\000\270.max_flat_workgroup_size\315\004\000\245.name\3317_Z25tqm_dequantize_kernel_tq4PK16block_tq4_mi300xPKfPfi\273.private_segment_fixed_size\000\253.sgpr_count\020\261.sgpr_spill_count\000\247.symbol\331:_Z25tqm_dequantize_kernel_tq4PK16block_tq4_mi300xPKfPfi.kd\270.uniform_work_group_size\001\263.uses_dynamic_stack\302\253.vgpr_countK\261.vgpr_spill_count\000\257.wavefront_size@\336\000\022\253.agpr_count\000\245.args\226\205\256.actual_access\251read_only\256.address_space\246global\247.offset\000\245.size\b\253.value_kind\255global_buffer\205\256.actual_access\251read_only\256.address_space\246global\247.offset\b\245.size\b\253.value_kind\255global_buffer\205\256.actual_access\251read_only\256.address_space\246global\247.offset\020\245.size\b\253.value_kind\255global_buffer\205\256.actual_access\251read_only\256.address_space\246global\247.offset\030\245.size\b\253.value_kind\255global_buffer\205\256.actual_access\252write_only\256.address_space\246global\247.offset \245.size\b\253.value_kind\255global_buffer\203\247.offset(\245.size\004\253.value_kind\250by_value\271.group_segment_fixed_size\315\002\f\266.kernarg_segment_align\b\265.kernarg_segment_size,\251.language\250OpenCL C\261.language_version\222\002\000\270.max_flat_workgroup_size\315\004\000\245.name\331C_Z14tqm_qjl_kernelPKfPK16block_tq3_mi300xS0_S0_P16block_qjl_mi300xi\273.private_segment_fixed_size\315\002\b\253.sgpr_count\023\261.sgpr_spill_count\000\247.symbol\331F_Z14tqm_qjl_kernelPKfPK16block_tq3_mi300xS0_S0_P16block_qjl_mi300xi.kd\270.uniform_work_group_size\001\263.uses_dynamic_stack\302\253.vgpr_count\314\200\261.vgpr_spill_count\314\201\257.wavefront_size@\255amdhsa.target\331)amdgcn-amd-amdhsa--gfx942:sramecc+:xnack-\256amdhsa.version\222\001\002\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\0007\000\000\000\021\003\006\000\300\034\000\000\000\000\000\000 \000\000\000\000\000\000\000\256\000\000\000\021\003\006\000\200\033\000\000\000\000\000\000@\000\000\000\000\000\000\000\230\001\000\000\021\003\006\000\000\034\000\000\000\000\000\000@\000\000\000\000\000\000\000\325\002\000\000\021\000\n\000p\243\000\000\000\000\000\000\001\000\000\000\000\000\000\000\001\000\000\000\022\003\007\000\000.\000\000\000\000\000\000\224\f\000\000\000\000\000\000\222\001\000\000\021\003\006\000\340\034\000\000\000\000\000\000@\000\000\000\000\000\000\000\321\001\000\000\022\003\007\000\000[\000\000\000\000\000\000\240\r\000\000\000\000\000\000D\002\000\000\022\003\007\000\000i\000\000\000\000\000\000\354\025\000\000\000\000\000\000=\000\000\000\021\003\006\000@\033\000\000\000\000\000\000@\000\000\000\000\000\000\000\351\000\000\000\022\003\007\000\000I\000\000\000\000\000\000\b\003\000\000\000\000\000\000\t\002\000\000\021\003\006\000@\034\000\000\000\000\000\000@\000\000\000\000\000\000\000\210\002\000\000\021\003\006\000\200\034\000\000\000\000\000\000@\000\000\000\000\000\000\000v\000\000\000\022\003\007\000\000;\000\000\000\000\000\000t\r\000\000\000\000\000\000!\001\000\000\021\003\006\000\300\033\000\000\000\000\000\000@\000\000\000\000\000\000\000\\\001\000\000\022\003\007\000\000M\000\000\000\000\000\000\200\r\000\000\000\000\000\000\317\002\000\000\021\003\006\000 \035\000\000\000\000\000\000\020\000\000\000\000\000\000\000\004\000\000\000\001\000\000\000\004\000\000\000\032\000\000\000\200\000!@\000 @\000(\215\b\000\002\000\000\202\013\000\004@\211\001\000\000\020\000 \000\000\004\000 \001\000\000\000\005\000\000\000\t\000\000\000\r\000\000\000\200\\F\017H\346(\347\240\000]K\021r\246U`e8\024\200\\F\017,\305\374x\325\244\377\364\236G\320\2006a\310\036\352#\234\023\223\302\002\243JtC*R/\027?\242\355-\234\177\\F\017\021\000\000\000\021\000\000\000\000\000\000\000\016\000\000\000\000\000\000\000\f\000\000\000\000\000\000\000\020\000\000\000\b\000\000\000\013\000\000\000\n\000\000\000\002\000\000\000\000\000\000\000\000\000\000\000\004\000\000\000\000\000\000\000\017\000\000\000\000\000\000\000\r\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\003\000\000\000\000\000\000\000\006\000\000\000\000\000\000\000\000\000\000\000\007\000\000\000\005\000\000\000\t\000\000\000\000_Z23tqm_quantize_kernel_tq3PKfS0_P16block_tq3_mi300xi\000d_cb3\000_Z23tqm_quantize_kernel_tq3PKfS0_P16block_tq3_mi300xi.kd\000_Z25tqm_dequantize_kernel_tq3PK16block_tq3_mi300xPKfPfi\000_Z25tqm_dequantize_kernel_tq3PK16block_tq3_mi300xPKfPfi.kd\000_Z24tqm_fused_dot_kernel_tq3PKfPK16block_tq3_mi300xPfii\000_Z24tqm_fused_dot_kernel_tq3PKfPK16block_tq3_mi300xPfii.kd\000_Z23tqm_quantize_kernel_tq4PKfS0_P16block_tq4_mi300xi\000d_cb4\000_Z23tqm_quantize_kernel_tq4PKfS0_P16block_tq4_mi300xi.kd\000_Z25tqm_dequantize_kernel_tq4PK16block_tq4_mi300xPKfPfi\000_Z25tqm_dequantize_kernel_tq4PK16block_tq4_mi300xPKfPfi.kd\000_Z14tqm_qjl_kernelPKfPK16block_tq3_mi300xS0_S0_P16block_qjl_mi300xi\000_Z14tqm_qjl_kernelPKfPK16block_tq3_mi300xS0_S0_P16block_qjl_mi300xi.kd\000d_cb2\000__hip_cuid_81ebb66926e4b9a5\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\f\002\000\000\204\002\000\000\034\000\000\000\000\000\000\000\300\022\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\037\000\000\000\217\000\257\000\205\000\000\000\b\000\000\000\000\000\000\000\004\002\000\000\000\000\000\000\034\000\000\000\000\000\000\000\200\037\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\022\000\000\000I\000\257\000\204\000\000\000\b\000\000\000\000\000\000\000\f\000\000\000\000\000\000\000 \000\000\000\000\000\000\000@-\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\002\000\000\000\201\000\257\000\204\001\000\000\b\000\000\000\000\000\000\000\f\002\000\000\204\002\000\000\034\000\000\000\000\000\000\000\0001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\037\000\000\000\317\000\257\000\205\000\000\000\b\000\000\000\000\000\000\000\004\002\000\000\000\000\000\000\034\000\000\000\000\000\000\000\300>\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\022\000\000\000I\000\257\000\204\000\000\000\b\000\000\000\000\000\000\000\f\002\000\000\b\002\000\000,\000\000\000\000\000\000\000\200L\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\037\000\000\000\217\000\257\000\205\000\000\000\b\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\300\000\002\300\030\000\000\000\177\300\214\277\002\003\003\277\037\003\205\277\000\001\006\300\000\000\000\000\000\002\006\300\020\000\000\000\002\000\375\321\002\016\001\004\237\004\006\"\177\300\214\277\002\000\b\322\002\005\021\000\000\200P\334\002\000\177\001\202\000\004$\377\002\006~\200\000\000\000p\017\214\277\000\000\032\330\002\001\000\000\177\300\214\277\000\000\212\277\000\000l\330\002\000\000\001\002\000\214\322\301\000\001\000\002\000\215\322\301\004\002\000\003\000\000\322\002\005\r\004\177\300\214\277\001\003\b\n\000\000~\330\003\004\000\003\277\004\b&\260\b\230}\177\300\214\277\001\003\006v\005\000\000\321\200 \251\001\005\000\376\321\005\005\n\002\000\000~\330\005\003\000\001\270\b\230}\177\300\214\277\003\003\002\002\005\000\000\321\200\020\251\001\005\000\376\321\005\005\n\002\000\000~\330\005\001\000\003\274\b\230}\177\300\214\277\001\007\002\002\005\000\000\321\200\b\251\001\005\000\376\321\005\005\n\002\000\000~\330\005\001\000\003\276\b\230}\177\300\214\277\001\007\002\002\005\000\000\321\200\004\251\001\005\000\376\321\005\005\n\002\000\000~\330\005\001\000\003\277\b\232}\001\000\200\277\200\004\b8\177\300\214\277\001\007\004\002\202\b\002$\000\000~\330\001\002\000\003\277\000\b&\206\000\002 \006\000\312\320\200\b\002\000\006 \204\276\005\000\210\277\202\002\b$\177\300\214\277\002\007\004\002\000\002\032\330\004\002\000\000~\004\376\207\200\002\004~\004\000\312\320\200\000\002\000\177\300\214\277\000\000\212\277\004 \212\276\006\000\210\277\000\002\354\330\002\000\000\004\177\300\214\277\005\t\006\002\b\002\032\330\002\003\000\000~\n\376\207\177\300\214\277\000\000\212\277\b\002l\330\002\000\000\002\002\264\203\226\002\264\002\222\b\002\002\200\377\000\210\276}\035\220&\177\300\214\277\002O\374~\t\003\003\202\b\374\206|\301\001\210\276\231\002\206\277\000\000\006\300\b\000\000\000\211\000\f$\200\002\376~\177\300\214\2770\200\\\334\006\000\000\n \200\\\334\006\000\000\002p\017\214\277P@|\334\000\002\177\000\020\200\\\334\006\000\000\002p\017\214\277\320@|\334\000\002\177\000\000\200\\\334\006\000\000\002p\017\214\277`@|\334\000\002\177\000\000\000\376\331\177\000\000\002\177\300\214\277p@|\334\000\002\177\000\020\000\376\331\177\000\000* \000\376\331\177\000\000\0020\000\376\331\177\000\000z\177\301\214\277 B|\334\000\002\177\000p\200\\\334\006\000\000 `\200\\\334\006\000\000\002p\017\214\277\200@|\334\000\002\177\000P\200\\\334\006\000\0006@\200\\\334\006\000\000\002p\017\214\277\320A|\334\000\002\177\000P\000\376\331\177\000\000>@\000\376\331\177\000\000\002\177\300\214\277\000B|\334\000\002\177\000p\000\376\331\177\000\000:`\000\376\331\177\000\000\002\177\300\214\277\220@|\334\000\002\177\000\260\200\\\334\006\000\000\002p\017\214\277PB|\334\000\002\177\000\240\200\\\334\006\000\000\002p\017\214\277\240@|\334\000\002\177\000\220\200\\\334\006\000\000F\200\200\\\334\006\000\000\002p\017\214\277@B|\334\000\002\177\000\220\000\376\331\177\000\000h\200\000\376\331\177\000\000\002\177\300\214\277`B|\334\000\002\177\000\260\000\376\331\177\000\000\002\177\300\214\277\240A|\334\000\002\177\000\240\000\376\331\177\000\000\002\177\300\214\277\260@|\334\000\002\177\000\360\200\\\334\006\000\000V\340\200\\\334\006\000\000\002p\017\214\277\000@|\334\000\002\177\000\320\200\\\334\006\000\000\032\300\200\\\334\006\000\000\002p\017\214\277pB|\334\000\002\177\000\320\000\376\331\177\000\000`\300\000\376\331\177\000\000\002\177\300\214\277\340A|\334\000\002\177\000\360\000\376\331\177\000\000\002\177\300\214\277\260A|\334\000\002\177\000\340\000\376\331\177\000\000\002\177\300\214\277\300@|\334\000\002\177\0000\201\\\334\006\000\000d \201\\\334\006\000\000\002p\017\214\277\020@|\334\000\002\177\000\020\201\\\334\006\000\000B\000\201\\\334\006\000\000\002p\017\214\277 @|\334\000\002\177\000\020\001\376\331\177\000\000Z\000\001\376\331\177\000\000\002\177\300\214\2770@|\334\000\002\177\0000\001\376\331\177\000\000R \001\376\331\177\000\000\002\177\300\214\277\340@|\334\000\002\177\000p\201\\\334\006\000\0002`\201\\\334\006\000\000\002p\017\214\277`A|\334\000\002\177\000P\201\\\334\006\000\000N@\201\\\334\006\000\000\002p\017\214\277\200A|\334\000\002\177\000P\001\376\331\177\000\000$@\001\376\331\177\000\000\002\177\300\214\277@@|\334\000\002\177\000p\001\376\331\177\000\000l`\001\376\331\177\000\000\002\177\300\214\277\220A|\334\000\002\177\000\260\201\\\334\006\000\000\002p\017\214\277\300A|\334\000\002\177\000\240\201\\\334\006\000\000\002p\017\214\277\000A|\334\000\002\177\000\220\201\\\334\006\000\000\016\200\201\\\334\006\000\000\002p\017\214\277\020A|\334\000\002\177\000\220\001\376\331\177\000\000.\200\001\376\331\177\000\000\002\177\300\214\277pA|\334\000\002\177\000\260\001\376\331\177\000\000\002\177\300\214\277\020B|\334\000\002\177\000\240\001\376\331\177\000\000\002\177\300\214\277PA|\334\000\002\177\000\360\201\\\334\006\000\000\002p\017\214\277\360A|\334\000\002\177\000\340\201\\\334\006\000\000\002p\017\214\277\360@|\334\000\002\177\000\320\201\\\334\006\000\000\024\300\201\\\334\006\000\000\002\000\034\200\276\000\377\000\200,\351\377\377\001\377\001\202\377\377\377\377\000\002\016\300\000\000\000\000p\017\214\277 A|\334\000\002\177\000\320\001\376\331\177\000\000J\300\001\376\331\177\000\000\002\177\300\214\277@A|\334\000\002\177\000\360\001\376\331\177\000\000\002\177\300\214\2770B|\334\000\002\177\000\340\001\376\331\177\000\000\002\177\300\214\2770A|\334\000\002\177\000\004 \200\276\002\000\210\277\000\200p\334\177~\002\000~\000\376\207\320@\\\334\000\000\177r\022@\261\323z\025\002\030hq\f~jq\020~\fq\364~p\017\214\277\n@\260\323*\345J\034\000\000\200\277\n@\260\323>m*\034bq\344~\n@\260\323:A*\034\240A\\\334\000\000\177hPB\\\334\000\000\177\036`q\340~\260A\\\334\000\000\177^\n@\260\323\006\215*\034Dq\214~Bq\210~&q|~q\017\214\277\n@\260\323h=*\034\034q<~\032q8~\n@\260\323p9*\034\020B\\\334\000\000\177\032q\017\214\277\n@\260\323^\255*\034\\q\274~Zq\270~Tq\254~\n@\260\323\\\211*\034Rq\250~Pq\244~\n@\260\323T\311*\034Nq\240~\n@\260\323$\241*\034\026q\210~\n@\260\323le*\0340q\330~\n@\260\323.\035*\034\300A\\\334\000\000\177.\024q\204~\360A\\\334\000\000\177N0B\\\334\000\000\177\026P@\\\334\000\000\177Z B\\\334\000\000\177\004`@\\\334\000\000\177(p@\\\334\000\000\177\022\200@\\\334\000\000\177b\020q\320~Rqt~\000B\\\334\000\000\177$x\017\214\277\n@\260\323\032]*\034\000\000\200\277\002@\260\323J\205*\034 q\204~v\017\214\277\002@\260\323\026\235\n\034t\017\214\277\006qd~\002@\260\323\004\265\n\034\320A\\\334\000\000\177\004s\017\214\277\002@\260\323\022Q\n\034\036q\224~q\017\214\277&qP~p\017\214\277\002@\260\323$\t\n\034\220@\\\334\000\000\177$\006ql~@B\\\334\000\000\177\004`B\\\334\000\000\177\020\240@\\\334\000\000\177vs\017\214\277\002@\260\323$\305\n\034r\017\214\277\006q\\~q\017\214\277\002@\260\323\020\t\n\034\022q,~\260@\\\334\000\000\177\020\340A\\\334\000\000\177\004pB\\\334\000\000\177Rq\017\214\277\006qH~\002@\260\323\020\355\n\034p\017\214\277Tq4~\002@\260\323\004\245\n\034\300@\\\334\000\000\177R\000@\\\334\000\000\177\004p\017\214\277\002@\260\323R\t\n\034 @\\\334\000\000\177\0040@\\\334\000\000\177\np\017\214\277\002@\260\323\n\t\n\034\340@\\\334\000\000\177\036\020@\\\334\000\000\177\004p\017\214\277\002@\260\323\036\t\n\034\200A\\\334\000\000\177\004@@\\\334\000\000\177\nq\017\214\277\006q\354~p\017\214\277\002@\260\323\n\t\n\034`A\\\334\000\000\177\004\220A\\\334\000\000\177\nq\017\214\277\006q\304~p\017\214\277\002@\260\323\n\t\n\034\fq\340~\020A\\\334\000\000\177\004pA\\\334\000\000\177\nq\017\214\277\006q\244~p\017\214\277\002@\260\323\n\t\n\034\fq\264~\000A\\\334\000\000\177\004PA\\\334\000\000\177\nq\017\214\277\006q<~p\017\214\277\002@\260\323\n\t\n\034\fq\234~ A\\\334\000\000\177\004@A\\\334\000\000\177\fq\017\214\277\006q ~p\017\214\277\002@\260\323\f\t\n\034\360@\\\334\000\000\177\0040A\\\334\000\000\177\np\017\214\277\002@\260\323\n\t\n\034\000\000\200\277\002@\260\323|\365\n\034~E\b~\002@\260\323,\351\n\034\203\002\374$\002@\260\323@q\n\034\000\000\200\277\002@\260\323<E\n\034\000\000\200\277\002@\260\323\b\221\n\034\000@\\\334\000\000\177\b\002@\260\323j\205\n\034\000\000\200\277\002@\260\323r\225\n\034\000\000\200\277\002@\260\323`\261\n\034\000\000\200\277\002@\260\323^\215\n\034\000\000\200\277\002@\260\323V\315\n\034\000\000\200\277\002@\260\323>u\n\034\000\000\200\277\002@\260\323ni\n\034\000\000\200\277\002@\260\323l\321\n\034\000\000\200\277\002@\260\323\034a\n\034\000\000\200\277\002@\260\323L\211\n\034\000\000\200\277\002@\260\323\030\241\n\034\000\000\200\277\002@\260\3232\271\n\034\000\000\200\277\002@\260\323\024U\n\034\000\000\200\277\002@\260\323(m\n\034\000\000\200\277\002@\260\323&\311\n\034\000\000\200\277\002@\260\323\026]\n\034\000\000\200\277\002@\260\323\022\361\n\034\000\000\200\277\002@\260\323$5\n\034p\017\214\277\002@\260\323T\025\n\034 @\\\334\000\000\177\b0@\\\334\000\000\177\022p\017\214\277\002@\260\323\024\025\n\034\020@\\\334\000\000\177\bp\017\214\277\002@\260\323 \025\n\034@@\\\334\000\000\177\bp\017\214\277\002@\260\323\n\355\n\034\000\000\200\277\002@\260\323p\305\n\034\000\000\200\277\002@\260\323Z\245\n\034\000\000\200\277\002@\260\323N=\n\034\000\000\200\277\002@\260\323\016!\n\034\000\000\200\277\002@\260\323\f\r\n\034\000\000\200\277\002\007\004\002\003\000\313\321\004\005\"\200\003\007\006\n\005\000\313\321\004\005&\200\377\006\006\024\312\362Iq\005\013\n\n\005\007\202|\007\000\313\321\004\005.\200\007\017\016\n\003\013\006\000\005\000\313\321\004\005*\200\005\013\n\n\005\007\f\024\t\000\313\321\004\0052\200\013\000\313\321\004\0056\200\r\000\313\321\004\005:\200\002\000\313\321\004\005>\200\004\000\000\321\200\002\251\001\005\007\214|\007\r\020\024\t\023\022\n\202\b\006\000\007\r\214|\t\021\024\024\013\027\026\n\203\006\006\000\t\021\214|\013\025\030\024\r\033\032\n\204\006\006\000\013\025\214|\r\031\034\024\002\005\004\n\205\006\006\000\r\031\214|\001\000\200\277\206\006\006\000\002\035\214|\001\000\200\277\207\006\b\000\201\b\002&\002\000\b\322\002\000\371\005\b\000\315\320\200\002\002\000\006 \200\276\003\000\210\277\bp\f~\004\200t\334\002\006\177\000~\000\376\207\001\000\310\321\004\003\005\002\b\000\315\320\200\002\002\000\006 \200\276\003\000\210\277\bp\f~\024\200t\334\002\006\177\000~\000\376\207\001\000\310\321\004\005\005\002\b\000\315\320\200\002\002\000\006 \200\276\003\000\210\277\bp\b~$\200t\334\002\004\177\000~\000\376\207\200\001\210\276~\b\352\206\016\000\206\277\004 \200\276\003\000\210\277\200\002\002~\000\200p\334\001\001\002\000~\000\376\207\206\000\230}j \200\276\005\000\210\277\203\000\004$\200\002\000~\000\003\002~\004\200t\334\002\000\002\000\000\000\201\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\300\000\002\300\030\000\000\000\177\300\214\277\002\003\003\277W\003\205\277\200\001\006\300\000\000\000\000\000\001\006\300\020\000\000\000\002\264\b\222\002\264\203\226\200\000\224}\177\300\214\277\006\b\006\200\007\003\007\202j \210\276\006\000\210\277\200\002\002~\000\200P\334\001\000\006\002p\017\214\277\000\002\032\330\001\002\000\000~\b\376\207\200\002\002~\177\300\214\277\000\000\212\277\000\002l\330\001\000\000\002\377\000\203\276}\035\220&\177\300\214\277\b\001A\320\002\007\000\000~\b\352\2061\003\207\277\203\000\002 \377\002\002&x\000\000\000\004\200T\334\001\000\006\002\024\200T\334\001\000\006\004$\200T\334\001\000\006\006\200\002\026~\000\034\206\276\006\377\006\200\004\341\377\377\007\377\007\202\377\377\377\377\000\000\006\300\b\000\000\000r\017\214\277\002\000\220\322\000\005\002\000q\017\214\277\004\000\220\322\000\t\002\000\201\004\002&p\017\214\277\006\000\220\322\000\r\002\000\203\b\b$\202\002\024$\204\f\n$\002\000\b\322\006\000)\004\210\b\024&\002\000\b\322\002\001)\004\220\n\024&\002\000\b\322\002\001)\004\000\200P\334\002\000\177\002\202\000\002$\377\002\006h\000\002\000\000\377\002\bh\000\004\000\000\377\002\nh\000\006\000\000\377\002\fh\000\b\000\000\377\002\016h\000\n\000\000\377\002\020h\000\f\000\000\377\002\022h\000\016\000\000\377\002\024h\0004\000\000\377\002>h\0006\000\000\377\002@h\0008\000\000\377\002Bh\000:\000\000\377\002Dh\000<\000\000\377\002Fh\000>\000\000\377\002X(\000@\000\000\377\002Zh\000B\000\000\377\002\\h\000D\000\000\377\002^h\000F\000\000\377\002bh\000J\000\000\377\002dh\000L\000\000\377\002fh\000N\000\000\377\002l(\000P\000\000\377\002nh\000R\000\000\377\002ph\000T\000\000\377\002rh\000V\000\000\377\002th\000X\000\000\377\002vh\000Z\000\000\377\002xh\000\\\000\000\377\002zh\000^\000\000\377\002|(\000`\000\000\377\002~h\000b\000\000\377\002\200h\000d\000\000\377\002\202h\000f\000\000\377\002\204(\000p\000\000\377\002\206h\000r\000\000\377\002\210h\000t\000\000\377\002\212h\000v\000\000\377\002\214(\000\200\000\000\377\002\216h\000\202\000\000\377\002\220h\000\204\000\000\377\002\222h\000\206\000\000\377\002\224h\000\210\000\000p\017\214\277\000\000\032\330\001\002\000\000\177\300\214\277\000\000\212\277\000\200P\334\001\000\000\032\000\200P\334\003\000\000\033\000\200P\334\004\000\000\034\000\200P\334\005\000\000\035\000\200P\334\006\000\000\030\000\200P\334\007\000\000\031\000\200P\334\b\000\000\026\000\200P\334\t\000\000\027\377\002\004(\000\020\000\000\377\002\006h\000\022\000\000\377\002\bh\000\024\000\000\000\200P\334\002\000\000\024\000\200P\334\003\000\000\025\377\002\nh\000\026\000\000\000\200P\334\004\000\000\020\000\200P\334\005\000\000\021\377\002\fh\000\030\000\000\377\002\004h\000\032\000\000\377\002\006h\000\034\000\000\377\002\bh\000\036\000\000\000\200P\334\006\000\000\016\000\200P\334\002\000\000\017\000\200P\334\003\000\000\f\000\200P\334\004\000\000\r\377\002\n(\000 \000\000\377\002\004h\000\"\000\000\377\002\006h\000$\000\000\000\200P\334\005\000\000\022\000\200P\334\002\000\000\023\377\002\004h\000&\000\000\000\200P\334\003\000\000\006\000\200P\334\002\000\000\007\377\002\004h\000(\000\000\377\002\006h\000*\000\000\377\002\bh\000,\000\000\377\002\nh\000.\000\000\000\200P\334\002\000\000\002\000\200P\334\003\000\000\003\000\200P\334\004\000\000\004\000\200P\334\005\000\000\005\377\002\020(\0000\000\000\377\002\022h\0002\000\000\000\200P\334\b\000\000\b\000\200P\334\t\000\000\t\000\200P\334\n\000\000\036\000\200P\334\037\000\000\037\000\000\376\331\013\000\000(\000\200P\334 \000\000&\000\200P\334!\000\000'\000\200P\334\"\000\000$\000\200P\334#\000\000%\000\200P\334,\000\000\"\000\200P\334-\000\000#\000\200P\334.\000\000 \000\200P\334/\000\000!\020\000\376\331\013\000\000,\377\002\024h\000H\000\000r\201\214\277\032@\261\323(5\002\030p\217\214\277\032@\260\323*9j\0340\000\376\331\013\000\000(~A\214\277\034@\260\323,1j\034 \000\376\331\013\000\000\030|O\214\277\026@\260\323.-r\034\377\002Xh\000h\000\000\377\002Zh\000j\000\000\377\002\\h\000l\000\000z@\214\277\024@\260\323\030)Z\034\377\002^h\000n\000\000xO\214\277\020@\260\323\032!R\034vO\214\277\024@\260\323(\035B\034@\000\376\331\013\000\000\016tO\214\277\f@\260\323*\031R\034P\000\376\331\013\000\000\024\000\200P\334\n\000\0000\000\200P\3341\000\0001\000\200P\3342\000\0004\000\200P\3343\000\0005\377\002Ph\000x\000\000vA\214\277\f@\260\323\016%2\034\377\002Rh\000z\000\000tO\214\277\006@\260\323\020\r2\034\377\002Th\000|\000\000r@\214\277\002@\260\323\024\005\032\034\377\002Vh\000~\000\000pO\214\277\006@\260\323\026\t\n\034`\000\376\331\013\000\000\002\000\200P\3346\000\0006\000\200P\3347\000\0007\000\200P\3348\000\000\030\000\200P\3349\000\000\031p\000\376\331\013\000\000\032\000\200P\334:\000\000\026\000\200P\334;\000\000\027\000\200P\334<\000\000\024\000\200P\334=\000\000\025\000\200P\334>\000\000\022\000\200P\334?\000\000\023\000\200P\334@\000\000\020\000\200P\334A\000\000\021zA\214\277\002@\260\323\002\021\032\034\377\002\024h\000\212\000\000xO\214\277\006@\260\323\004=\n\034\000\200P\334,\000\000\016\000\200P\334-\000\000\017\000\200P\334.\000\000\f\000\200P\334/\000\000\r\000\200P\334B\000\000\004\000\200P\334C\000\000\005\000\200P\334D\000\000\002\000\200P\334E\000\000\003~@\214\277\032@\260\323\032M\032\034\200\000\376\331\013\000\000\006\220\000\376\331\013\000\000,|O\214\277\032@\260\323\034Ij\034\377\002<(\000\220\000\000\377\002>h\000\222\000\000\377\002Lh\000\230\000\000\377\002Nh\000\232\000\000zA\214\277\006@\260\323\006Ej\034\377\002ph\000\214\000\000\377\002rh\000\216\000\000\377\002th\000\224\000\000\377\002vh\000\226\000\000\377\002xh\000\234\000\000\377\002zh\000\236\000\000xO\214\2772@\260\323\bA\032\034\377\002|(\000\240\000\000\377\002~h\000\242\000\000\000\200P\334(\000\000\034\000\200P\334)\000\000\035\000\200P\334*\000\000\b\000\200P\334+\000\000\t\000\200P\334F\000\000\"\000\200P\334G\000\000#\000\200P\334H\000\000\006\000\200P\334I\000\000\007\000\200P\334J\000\000 \000\200P\334\n\000\000!\000\200P\3348\000\000\032\000\200P\3349\000\000\033\000\200P\334\036\000\000\036\000\200P\334\037\000\000\037\000\200P\334:\000\000$\000\200P\334;\000\000%\000\200P\334&\000\000&\000\200P\334'\000\000'\000\200P\334<\000\000(\000\200P\334=\000\000)\000\200P\334>\000\000*\000\200P\334?\000\000+\377\002\024h\000\244\000\000\377\002ph\000\246\000\000\377\002rh\000\250\000\000\377\002th\000\252\000\000\377\002vh\000\270\000\000\377\002xh\000\272\000\000\377\002z(\000\300\000\000\377\002|h\000\302\000\000\377\002~h\000\304\000\000\377\002\200h\000\306\000\000\377\002\202h\000\310\000\000\377\002\204h\000\312\000\000\377\002\206h\000\314\000\000\377\002\210h\000\316\000\000|\200\214\277,@\260\323,a\312\034\240\000\376\331\013\000\0000z\217\214\2774@\260\323.i\262\034\260\000\376\331\013\000\000,x\201\214\2770@\260\3230m\322\034v\217\214\277\030@\260\32321\302\034\377\002d(\000\260\000\000t\200\214\277,@\260\323,-b\034\300\000\376\331\013\000\000\026r\217\214\277\024@\260\323.)\262\034\320\000\376\331\013\000\000,\377\002fh\000\262\000\000\377\002hh\000\254\000\000p\201\214\277\022@\260\323\026%R\034\377\002,h\000\274\000\000~O\214\277\020@\260\323\030!J\034\377\002.h\000\276\000\000|@\214\277\022@\260\323,\035B\034\340\000\376\331\013\000\000\016zO\214\277\f@\260\323.\031J\034\360\000\376\331\013\000\000,\377\002jh\000\256\000\000\377\002lh\000\264\000\000xA\214\277\004@\260\323\016\t2\034\377\002nh\000\266\000\000vO\214\2770@\260\323\020\005\022\034\000\001\376\331\013\000\000\002\000\200P\334\n\000\000\020\000\200P\3348\000\000\021\000\200P\3349\000\000\016\000\200P\334:\000\000\017\000\200P\3344\000\000\f\000\200P\3345\000\000\r\000\200P\3342\000\000\024\000\200P\3343\000\000\025\000\200P\3346\000\000\022\000\200P\3347\000\000\023\000\200P\334;\000\000\030\000\200P\334<\000\000\031\000\200P\334\026\000\000\026\000\200P\334\027\000\000\027r\201\214\277\034@\260\323,9\302\034\020\001\376\331\013\000\0000p\217\214\277\b@\260\323.\021r\034 \001\376\331\013\000\000,0\001\376\331\013\000\0004~C\214\277\002@\260\323\002E\"\034\377\002\024(\000\320\000\000|O\214\277\002@\260\323\004\r\n\034\377\002Dh\000\322\000\000zB\214\277\034@\260\3230A\n\034@\001\376\331\013\000\000\006P\001\376\331\013\000\000\002xO\214\277\032@\260\32325r\034\377\002Fh\000\324\000\000vC\214\277\032@\260\323,=j\034\377\002`h\000\350\000\000tO\214\277\032@\260\323.Ij\034\377\002^h\000\346\000\000rB\214\277\032@\260\3234Mj\034\377\002Nh\000\326\000\000pO\214\277\032@\260\3236Qj\034\377\002Ph\000\330\000\000~\001\214\277 @\260\323\006Uj\034\000\200P\334=\000\000\032\000\200P\334>\000\000\033\000\200P\334?\000\000\006\000\200P\334@\000\000\007\000\200P\334A\000\000\036\000\200P\334B\000\000\037\000\200P\334C\000\000\034\000\200P\334D\000\000\035\377\002Rh\000\332\000\000\377\002Th\000\334\000\000\377\002Vh\000\336\000\000\000\200P\334\n\000\000$\000\200P\334\"\000\000%\000\200P\334#\000\000&\000\200P\334'\000\000'\000\200P\334(\000\000(\000\200P\334)\000\000)\000\200P\334*\000\000*\000\200P\334+\000\000+\377\002\024(\000\340\000\000\377\002bh\000\352\000\000\377\002dh\000\354\000\000\377\002fh\000\356\000\000\377\002Dh\000\342\000\000\377\002Fh\000\344\000\000\000\200P\334\n\000\000,\000\200P\334\"\000\000-\000\200P\334#\000\000.\000\200P\334/\000\000/\000\200P\3340\000\0000\000\200P\3341\000\0001\000\200P\3342\000\0002\000\200P\3343\000\0003\377\002\024(\000\360\000\000\377\002nh\000\366\000\000\377\002ph\000\370\000\000\377\002rh\000\372\000\000\377\002th\000\374\000\000\377\002Dh\000\362\000\000\377\002Fh\000\364\000\000\377\002\002h\000\376\000\000\000\200P\334\n\000\0004\000\200P\334\"\000\0005\000\200P\334#\000\0006\000\200P\3347\000\0007\000\200P\3348\000\0008\000\200P\3349\000\0009\000\200P\334:\000\000:\000\200P\334\001\000\000;\000\002l\330\013\000\000\001|\217\214\277\b@\260\323\b!\202\034`\001\376\331\013\000\000 z\202\214\277\002@\260\323\002\035\"\034p\001\376\331\013\000\000\016x\217\214\277\002@\260\323\004\031\n\034v\201\214\277\b@\260\323 )\n\034\200\001\376\331\013\000\000\002t\217\214\277\b@\260\323\"%\"\034r\201\214\277\b@\260\323\0161\"\034\220\001\376\331\013\000\000\fp\217\214\277\b@\260\323\020-\"\034\240\001\376\331\013\000\000\020~B\214\277\002@\260\323\0025\"\034|O\214\277\002@\260\323\004\r\n\034zA\214\277\006@\260\323\f=\n\034\260\001\376\331\013\000\000\002xO\214\277\006@\260\323\0169\032\034vA\214\277\f@\260\323\020I\032\034\300\001\376\331\013\000\000\006tO\214\277\f@\260\323\022M2\034rA\214\277\002@\260\323\002Q2\034\320\001\376\331\013\000\000\fpO\214\277\002@\260\323\004U\n\034~\001\214\277\006@\260\323\006Y\n\034\340\001\376\331\013\000\000\002|\017\214\277\006@\260\323\b]\032\034z\001\214\277\f@\260\323\fa\032\034\360\001\376\331\013\000\000\006x\017\214\277\f@\260\323\016e2\034v\001\214\277\002@\260\323\002i2\034t\017\214\277\002@\260\323\004m\n\034r\000\214\277\002@\260\323\006q\n\034p\017\214\277\002@\260\323\bu\n\034\000\000\200\277\002\007\004\002\001\005\002\n\002\000\375\321\002\016\001\004\237\004\006\"\002\000\b\322\002\005\021\000\000\200p\334\002\001\177\000\000\000\201\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\200\001\006\300\030\000\000\000\177\300\214\277\002\007\003\277\301\200\204\205\003\006\003\277\301\200\210\205\b\004\204\207~\004\352\206\267\000\207\277\000\001\006\300\b\000\000\000\002\264\b\222\002\264\206\226\177\300\214\277\004\b\b\200\005\006\t\202\004\000\312\320\200\000\002\000\004 \212\276\006\000\210\277\200\002\002~\000\200P\334\001\000\b\002p\017\214\277\b\000\032\330\001\002\000\000~\n\376\207\200\002\006~\177\300\214\277\000\000\212\277\b\000l\330\003\000\000\001\377\000\206\276}\035\220&\177\300\214\277\n\001A\320\001\r\000\000~\n\352\206\205\000\207\277\206\000\002 \203\002\004$\004\200T\334\002\000\b\004\024\200T\334\002\000\b\006$\200T\334\002\000\b\b\000\002\006\300\000\000\000\000\000\034\212\276\n\377\n\200\364\322\377\377\013\377\013\202\377\377\377\377\n\000\375\321\003\016\001\004\237\024\026\"\177\300\214\277\n\000\b\322\n\005!\000\000\200P\334\n\000\177\ns\017\214\277\004\000\220\322\000\t\002\000r\017\214\277\006\000\220\322\000\r\002\000\201\b\004&q\017\214\277\b\000\220\322\000\021\002\000\203\f\f$\202\004\004$\204\020\016$\004\000\b\322\n\000\t\004\210\f\004&\004\000\b\322\004\001\t\004\220\016\004&\002\000\b\322\004\001\t\004\000\200P\334\002\000\177\002\003\000\214\322\301\000\001\000\377\002\b~\200\000\000\000\003\000\215\322\301\006\002\000\004\000\000\322\003\005\021\004\277\000\000&p\017\214\277\n\005\n\n\000\000~\330\004\005\000\004\277\006\n&\260\n\230}\177\300\214\277\n\005\bv\006\000\000\321\200 \251\001\006\000\376\321\006\007\n\002\000\000~\330\006\004\000\002\270\n\230}\177\300\214\277\004\005\004\002\006\000\000\321\200\020\251\001\006\000\376\321\006\007\n\002\000\000~\330\006\002\000\004\274\n\230}\177\300\214\277\002\t\004\002\006\000\000\321\200\b\251\001\006\000\376\321\006\007\n\002\000\000~\330\006\002\000\004\276\n\230}\177\300\214\277\002\t\004\002\006\000\000\321\200\004\251\001\006\000\376\321\006\007\n\002\000\000~\330\006\002\000\004\277\n\232}\177\300\214\277\002\t\004\002\200\006\0068\202\006\006$\000\000~\330\003\002\000\003\200\000\224}j \210\276\005\000\210\277\202\002\000$\177\300\214\277\002\007\002\002\000\000\032\330\000\001\000\000~\b\376\207\200\001\212\276\200\001\210\276\177\300\214\277\000\000\212\277\004 \214\276~\f\214\210\n\000\210\277\200\002\004~\000\000\354\330\002\000\000\000\b\000l\330\002\000\000\002~\001\210\276\177\301\214\277\001\001\000\002\177\300\214\277\000\005\000\n~\f\376\207~\n\352\206\003\000\207\277\006\000\202\277\200\001\210\276\004\000\210\277\b~\210\211\004~\204\206\200\002\000~\b\004\210\207\b \204\276\f\000\210\277\000\000\006\300\020\000\000\000\007\003\003\222\003\002\002\201\002\237\003\220\002\202\202\216\177\300\214\277\000\002\000\200\001\003\001\202\200\002\002~\000\200p\334\001\000\000\000\000\000\201\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\300\000\002\300\030\000\000\000\177\300\214\277\002\003\003\277Z\003\205\277\000\001\006\300\000\000\000\000\000\002\006\300\020\000\000\000\002\000\375\321\002\016\001\004\237\004\006\"\177\300\214\277\002\000\b\322\002\005\021\000\000\200P\334\002\000\177\001\202\000\004$\377\002\006~\200\000\000\000p\017\214\277\000\000\032\330\002\001\000\000\177\300\214\277\000\000\212\277\000\000l\330\002\000\000\001\002\000\214\322\301\000\001\000\002\000\215\322\301\004\002\000\003\000\000\322\002\005\r\004\177\300\214\277\001\003\b\n\000\000~\330\003\004\000\003\277\004\b&\260\b\230}\177\300\214\277\001\003\006v\005\000\000\321\200 \251\001\005\000\376\321\005\005\n\002\000\000~\330\005\003\000\001\270\b\230}\177\300\214\277\003\003\002\002\005\000\000\321\200\020\251\001\005\000\376\321\005\005\n\002\000\000~\330\005\001\000\003\274\b\230}\177\300\214\277\001\007\002\002\005\000\000\321\200\b\251\001\005\000\376\321\005\005\n\002\000\000~\330\005\001\000\003\276\b\230}\177\300\214\277\001\007\002\002\005\000\000\321\200\004\251\001\005\000\376\321\005\005\n\002\000\000~\330\005\001\000\003\277\b\232}\001\000\200\277\200\004\b8\177\300\214\277\001\007\004\002\202\b\002$\000\000~\330\001\002\000\003\277\000\b&\206\000\002 \006\000\312\320\200\b\002\000\006 \204\276\005\000\210\277\202\002\b$\177\300\214\277\002\007\004\002\000\002\032\330\004\002\000\000~\004\376\207\200\002\004~\004\000\312\320\200\000\002\000\177\300\214\277\000\000\212\277\004 \212\276\006\000\210\277\000\002\354\330\002\000\000\004\177\300\214\277\005\t\006\002\b\002\032\330\002\003\000\000~\n\376\207\177\300\214\277\000\000\212\277\b\002l\330\002\000\000\002\002\377\203\226D\000\000\000D\000\202\267\b\002\002\200\377\000\210\276}\035\220&\177\300\214\277\002O\374~\t\003\003\202\b\374\206|\301\001\210\276\323\002\206\277\000\000\006\300\b\000\000\000\211\000\f$\200\002\376~\177\300\214\2770\200\\\334\006\000\000\n \200\\\334\006\000\000\002p\017\214\277P@|\334\000\002\177\000\020\200\\\334\006\000\000\002p\017\214\277\320@|\334\000\002\177\000\000\200\\\334\006\000\000\002p\017\214\277`@|\334\000\002\177\000\000\000\376\331\177\000\000\002\177\300\214\277p@|\334\000\002\177\000\020\000\376\331\177\000\000* \000\376\331\177\000\000\0020\000\376\331\177\000\000z\177\301\214\277 B|\334\000\002\177\000p\200\\\334\006\000\000 `\200\\\334\006\000\000\002p\017\214\277\200@|\334\000\002\177\000P\200\\\334\006\000\0006@\200\\\334\006\000\000\002p\017\214\277\320A|\334\000\002\177\000P\000\376\331\177\000\000>@\000\376\331\177\000\000\002\177\300\214\277\000B|\334\000\002\177\000p\000\376\331\177\000\000:`\000\376\331\177\000\000\002\177\300\214\277\220@|\334\000\002\177\000\260\200\\\334\006\000\000\002p\017\214\277PB|\334\000\002\177\000\240\200\\\334\006\000\000\002p\017\214\277\240@|\334\000\002\177\000\220\200\\\334\006\000\000F\200\200\\\334\006\000\000\002p\017\214\277@B|\334\000\002\177\000\220\000\376\331\177\000\000h\200\000\376\331\177\000\000\002\177\300\214\277`B|\334\000\002\177\000\260\000\376\331\177\000\000\002\177\300\214\277\240A|\334\000\002\177\000\240\000\376\331\177\000\000\002\177\300\214\277\260@|\334\000\002\177\000\360\200\\\334\006\000\000V\340\200\\\334\006\000\000\002p\017\214\277\000@|\334\000\002\177\000\320\200\\\334\006\000\000\032\300\200\\\334\006\000\000\002p\017\214\277pB|\334\000\002\177\000\320\000\376\331\177\000\000`\300\000\376\331\177\000\000\002\177\300\214\277\340A|\334\000\002\177\000\360\000\376\331\177\000\000\002\177\300\214\277\260A|\334\000\002\177\000\340\000\376\331\177\000\000\002\177\300\214\277\300@|\334\000\002\177\0000\201\\\334\006\000\000d \201\\\334\006\000\000\002p\017\214\277\020@|\334\000\002\177\000\020\201\\\334\006\000\000B\000\201\\\334\006\000\000\002p\017\214\277 @|\334\000\002\177\000\020\001\376\331\177\000\000Z\000\001\376\331\177\000\000\002\177\300\214\2770@|\334\000\002\177\0000\001\376\331\177\000\000R \001\376\331\177\000\000\002\177\300\214\277\340@|\334\000\002\177\000p\201\\\334\006\000\0002`\201\\\334\006\000\000\002p\017\214\277`A|\334\000\002\177\000P\201\\\334\006\000\000N@\201\\\334\006\000\000\002p\017\214\277\200A|\334\000\002\177\000P\001\376\331\177\000\000$@\001\376\331\177\000\000\002\177\300\214\277@@|\334\000\002\177\000p\001\376\331\177\000\000l`\001\376\331\177\000\000\002\177\300\214\277\220A|\334\000\002\177\000\260\201\\\334\006\000\000\002p\017\214\277\300A|\334\000\002\177\000\240\201\\\334\006\000\000\002p\017\214\277\000A|\334\000\002\177\000\220\201\\\334\006\000\000\016\200\201\\\334\006\000\000\002p\017\214\277\020A|\334\000\002\177\000\220\001\376\331\177\000\000.\200\001\376\331\177\000\000\002\177\300\214\277pA|\334\000\002\177\000\260\001\376\331\177\000\000\002\177\300\214\277\020B|\334\000\002\177\000\240\001\376\331\177\000\000\002\177\300\214\277PA|\334\000\002\177\000\360\201\\\334\006\000\000\002p\017\214\277\360A|\334\000\002\177\000\340\201\\\334\006\000\000\002p\017\214\277\360@|\334\000\002\177\000\320\201\\\334\006\000\000\024\300\201\\\334\006\000\000\002\000\034\200\276\000\377\000\200H\312\377\377\001\377\001\202\377\377\377\377\000\002\022\300\000\000\000\000p\017\214\277 A|\334\000\002\177\000\320\001\376\331\177\000\000J\300\001\376\331\177\000\000\002\177\300\214\277@A|\334\000\002\177\000\360\001\376\331\177\000\000\002\177\300\214\2770B|\334\000\002\177\000\340\001\376\331\177\000\000\002\177\300\214\2770A|\334\000\002\177\000\004 \200\276\002\000\210\277\000\200p\334\177~\002\000~\000\376\207\320@\\\334\000\000\177r\022@\261\323z\025\002\030hq\f~jq\020~\fq\364~p\017\214\277\n@\260\323*\345J\034\000\000\200\277\n@\260\323>m*\034bq\344~\n@\260\323:A*\034\240A\\\334\000\000\177hPB\\\334\000\000\177\036`q\340~\260A\\\334\000\000\177^\n@\260\323\006\215*\034Dq\214~Bq\210~&q|~q\017\214\277\n@\260\323h=*\034\034q<~\032q8~\n@\260\323p9*\034\020B\\\334\000\000\177\032q\017\214\277\n@\260\323^\255*\034\\q\274~Zq\270~Tq\254~\n@\260\323\\\211*\034Rq\250~Pq\244~\n@\260\323T\311*\034Nq\240~\n@\260\323$\241*\034\026q\210~\n@\260\323le*\0340q\330~\n@\260\323.\035*\034\300A\\\334\000\000\177.\024q\204~\360A\\\334\000\000\177N0B\\\334\000\000\177\026P@\\\334\000\000\177Z B\\\334\000\000\177\004`@\\\334\000\000\177(p@\\\334\000\000\177\022\200@\\\334\000\000\177b\020q\320~Rqt~\000B\\\334\000\000\177$x\017\214\277\n@\260\323\032]*\034\000\000\200\277\002@\260\323J\205*\034 q\204~v\017\214\277\002@\260\323\026\235\n\034t\017\214\277\006qd~\002@\260\323\004\265\n\034\320A\\\334\000\000\177\004s\017\214\277\002@\260\323\022Q\n\034\036q\224~q\017\214\277&qP~p\017\214\277\002@\260\323$\t\n\034\220@\\\334\000\000\177$\006ql~@B\\\334\000\000\177\004`B\\\334\000\000\177\020\240@\\\334\000\000\177vs\017\214\277\002@\260\323$\305\n\034r\017\214\277\006q\\~q\017\214\277\002@\260\323\020\t\n\034\022q,~\260@\\\334\000\000\177\020\340A\\\334\000\000\177\004pB\\\334\000\000\177Rq\017\214\277\006qH~\002@\260\323\020\355\n\034p\017\214\277Tq4~\002@\260\323\004\245\n\034\300@\\\334\000\000\177R\000@\\\334\000\000\177\004p\017\214\277\002@\260\323R\t\n\034 @\\\334\000\000\177\0040@\\\334\000\000\177\np\017\214\277\002@\260\323\n\t\n\034\340@\\\334\000\000\177\036\020@\\\334\000\000\177\004p\017\214\277\002@\260\323\036\t\n\034\200A\\\334\000\000\177\004@@\\\334\000\000\177\nq\017\214\277\006q\354~p\017\214\277\002@\260\323\n\t\n\034`A\\\334\000\000\177\004\220A\\\334\000\000\177\nq\017\214\277\006q\304~p\017\214\277\002@\260\323\n\t\n\034\fq\340~\020A\\\334\000\000\177\004pA\\\334\000\000\177\nq\017\214\277\006q\244~p\017\214\277\002@\260\323\n\t\n\034\fq\264~\000A\\\334\000\000\177\004PA\\\334\000\000\177\nq\017\214\277\006q<~p\017\214\277\002@\260\323\n\t\n\034\fq\234~ A\\\334\000\000\177\004@A\\\334\000\000\177\fq\017\214\277\006q ~p\017\214\277\002@\260\323\f\t\n\034\360@\\\334\000\000\177\0040A\\\334\000\000\177\np\017\214\277\002@\260\323\n\t\n\034\000\000\200\277\002@\260\323|\365\n\034~E\b~\002@\260\323,\351\n\034\203\002\374$\002@\260\323@q\n\034\000\000\200\277\002@\260\323<E\n\034\000\000\200\277\002@\260\323\b\221\n\034\000@\\\334\000\000\177\b\002@\260\323j\205\n\034\000\000\200\277\002@\260\323r\225\n\034\000\000\200\277\002@\260\323`\261\n\034\000\000\200\277\002@\260\323^\215\n\034\000\000\200\277\002@\260\323V\315\n\034\000\000\200\277\002@\260\323>u\n\034\000\000\200\277\002@\260\323ni\n\034\000\000\200\277\002@\260\323l\321\n\034\000\000\200\277\002@\260\323\034a\n\034\000\000\200\277\002@\260\323L\211\n\034\000\000\200\277\002@\260\323\030\241\n\034\000\000\200\277\002@\260\3232\271\n\034\000\000\200\277\002@\260\323\024U\n\034\000\000\200\277\002@\260\323(m\n\034\000\000\200\277\002@\260\323&\311\n\034\000\000\200\277\002@\260\323\026]\n\034\000\000\200\277\002@\260\323\022\361\n\034\000\000\200\277\002@\260\323$5\n\034p\017\214\277\002@\260\323T\025\n\034 @\\\334\000\000\177\b0@\\\334\000\000\177\022p\017\214\277\002@\260\323\024\025\n\034\020@\\\334\000\000\177\bp\017\214\277\002@\260\323 \025\n\034@@\\\334\000\000\177\bp\017\214\277\002@\260\323\n\355\n\034\000\000\200\277\002@\260\323p\305\n\034\000\000\200\277\002@\260\323Z\245\n\034\000\000\200\277\002@\260\323N=\n\034\000\000\200\277\002@\260\323\016!\n\034\000\000\200\277\002@\260\323\f\r\n\034\000\000\200\277\002\007\004\002\003\000\313\321\004\005\"\200\003\007\006\n\005\000\313\321\004\005&\200\377\006\006\024\312\362Iq\005\013\n\n\005\007\202|\007\000\313\321\004\005.\200\007\017\016\n\003\013\006\000\005\000\313\321\004\005*\200\005\013\n\n\005\007\f\024\t\000\313\321\004\0052\200\013\000\313\321\004\0056\200\r\000\313\321\004\005:\200\017\000\313\321\004\005>\200\021\000\313\321\004\005B\200\023\000\313\321\004\005F\200\025\000\313\321\004\005J\200\027\000\313\321\004\005N\200\031\000\313\321\004\005R\200\033\000\313\321\004\005V\200\035\000\313\321\004\005Z\200\002\000\313\321\004\005^\200\004\000\000\321\200\002\251\001\005\007\214|\007\r\020\024\t\023\022\n\202\b\006\000\007\r\214|\t\021\024\024\013\027\026\n\203\006\006\000\t\021\214|\013\025\030\024\r\033\032\n\204\006\006\000\013\025\214|\r\031\034\024\017\037\036\n\205\006\006\000\r\031\214|\017\035 \024\021#\"\n\206\006\006\000\017\035\214|\021!$\024\023'&\n\207\006\006\000\021!\214|\023%(\024\025+*\n\210\006\006\000\023%\214|\025),\024\027/.\n\211\006\006\000\025)\214|\027-0\024\03132\n\212\006\006\000\027-\214|\03114\024\03376\n\213\006\006\000\0311\214|\03358\024\035;:\n\214\006\006\000\0335\214|\0359<\024\002\005\004\n\215\006\006\000\0359\214|\001\000\200\277\216\006\006\000\002=\214|\001\000\200\277\217\006\b\000\201\b\002&\002\000\b\322\002\000\371\005\b\000\315\320\200\002\002\000\006 \200\276\003\000\210\277\bp\f~\004\200t\334\002\006\177\000~\000\376\207\001\000\310\321\004\003\005\002\b\000\315\320\200\002\002\000\006 \200\276\003\000\210\277\bp\f~\024\200t\334\002\006\177\000~\000\376\207\001\000\310\321\004\005\005\002\b\000\315\320\200\002\002\000\006 \200\276\003\000\210\277\bp\f~$\200t\334\002\006\177\000~\000\376\207\001\000\310\321\004\007\005\002\b\000\315\320\200\002\002\000\006 \200\276\003\000\210\277\bp\b~4\200t\334\002\004\177\000~\000\376\207\200\001\210\276~\b\352\206\016\000\206\277\004 \200\276\003\000\210\277\200\002\002~\000\200p\334\001\001\002\000~\000\376\207\210\000\230}j \200\276\005\000\210\277\203\000\004$\200\002\000~\000\003\002~\004\200t\334\002\000\002\000\000\000\201\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\300\000\002\300\030\000\000\000\177\300\214\277\002\003\003\277b\003\205\277\200\001\006\300\000\000\000\000\000\001\006\300\020\000\000\000\002\377\b\222D\000\000\000\002\377\203\226D\000\000\000\200\000\224}\177\300\214\277\006\b\006\200\007\003\007\202j \210\276\006\000\210\277\200\002\002~\000\200P\334\001\000\006\002p\017\214\277\000\002\032\330\001\002\000\000~\b\376\207\200\002\002~\177\300\214\277\000\000\212\277\000\002l\330\001\000\000\002\377\000\203\276}\035\220&\177\300\214\277\b\001A\320\002\007\000\000~\b\352\206:\003\207\277\203\000\002 \377\002\002&x\000\000\000\004\200T\334\001\000\006\002\024\200T\334\001\000\006\004$\200T\334\001\000\006\0064\200T\334\001\000\006\b\200\002\026~\000\034\206\276\006\377\006\200\024\301\377\377\007\377\007\202\377\377\377\377\000\000\006\300\b\000\000\000s\017\214\277\002\000\220\322\000\005\002\000r\017\214\277\004\000\220\322\000\t\002\000\201\004\002&q\017\214\277\006\000\220\322\000\r\002\000\203\b\b$\202\002\024$p\017\214\277\b\000\220\322\000\021\002\000\204\f\n$\002\000\b\322\006\000)\004\210\b\024&\205\020\f$\002\000\b\322\002\001)\004\220\n\024&\002\000\b\322\002\001)\004\240\f\024&\002\000\b\322\002\001)\004\000\200P\334\002\000\177\002\202\000\002$\377\002\006h\000\002\000\000\377\002\bh\000\004\000\000\377\002\nh\000\006\000\000\377\002\f(\000\020\000\000\377\002\016h\0002\000\000\377\002\024h\0004\000\000\377\002Ph\0008\000\000\377\002Rh\000:\000\000\377\002Th\000>\000\000\377\002V(\000@\000\000\377\002Xh\000B\000\000\377\002Zh\000D\000\000\377\002\\h\000F\000\000\377\002^h\000H\000\000\377\002`h\000J\000\000\377\002bh\000L\000\000\377\002dh\000N\000\000\377\002f(\000P\000\000\377\002hh\000R\000\000\377\002jh\000T\000\000\377\002lh\000V\000\000\377\002nh\000X\000\000\377\002ph\000Z\000\000\377\002rh\000\\\000\000\377\002th\000^\000\000\377\002v(\000`\000\000\377\002xh\000b\000\000\377\002zh\000d\000\000\377\002|h\000f\000\000\377\002~h\000h\000\000\377\002\200h\000v\000\000\377\002\202(\000\200\000\000\377\002\204h\000\202\000\000\377\002\206h\000\204\000\000\377\002\210h\000\206\000\000\377\002\212h\000\210\000\000\377\002\214h\000\212\000\000\377\002\216h\000\216\000\000\377\002\220(\000\220\000\000\377\002\222h\000\222\000\000\377\002\224h\000\224\000\000p\017\214\277\000\000\032\330\001\002\000\000\177\300\214\277\000\000\212\277\000\200P\334\001\000\000\032\000\200P\334\003\000\000\033\000\200P\334\004\000\000\034\000\200P\334\005\000\000\035\377\002\004h\000\b\000\000\377\002\006h\000\n\000\000\377\002\bh\000\f\000\000\377\002\nh\000\016\000\000\000\200P\334\002\000\000\016\000\200P\334\003\000\000\017\000\200P\334\004\000\000\020\000\200P\334\005\000\000\021\377\002\004h\000\022\000\000\377\002\006h\000\024\000\000\000\200P\334\006\000\000\036\000\200P\334\002\000\000\037\377\002\bh\000\026\000\000\000\200P\334\003\000\000\026\000\200P\334\004\000\000\027\377\002\nh\000\030\000\000\377\002\004h\000\032\000\000\377\002\006h\000\034\000\000\377\002\bh\000\036\000\000\000\200P\334\005\000\000\024\000\200P\334\002\000\000\025\000\200P\334\003\000\000\022\000\200P\334\004\000\000\023\377\002\f(\000 \000\000\377\002\004h\000\"\000\000\377\002\006h\000$\000\000\000\200P\334\006\000\000\030\000\200P\334\002\000\000\031\377\002\004h\000&\000\000\000\200P\334\003\000\000\f\000\200P\334\002\000\000\r\377\002\004h\000(\000\000\377\002\006h\000*\000\000\377\002\bh\000,\000\000\377\002\nh\000.\000\000\000\200P\334\002\000\000\002\000\200P\334\003\000\000\003\000\200P\334\004\000\000\004\000\200P\334\005\000\000\005\377\002\f(\0000\000\000\000\200P\334\006\000\000\b\000\200P\334\007\000\000\t\377\002\016h\0006\000\000\000\200P\334\n\000\000\006\000\200P\334\007\000\000\007\000\000\376\331\013\000\000 \020\000\376\331\013\000\000$\377\002\024h\000<\000\000zA\214\277\032@\261\323 5\002\030xO\214\277\"@\260\323\"9j\034\000\200P\334(\000\000 \000\200P\334)\000\000!\000\200P\334\n\000\000\034\000\200P\334*\000\000\035\000\200P\334+\000\000\032\000\200P\334,\000\000\033\377\002\024h\000j\000\000|@\214\277\016@\260\323$\035\212\034 \000\376\331\013\000\000\"zO\214\277\020@\260\323&!:\034\000\200P\334-\000\000\016\000\200P\334.\000\000\0170\000\376\331\013\000\000&\377\002Th\000l\000\000zA\214\277\020@\260\323\"=B\034\377\002Vh\000n\000\000xO\214\277\020@\260\323$-B\034\377\002X(\000p\000\000v@\214\277\020@\260\323&)B\034@\000\376\331\013\000\000\024tO\214\277\036@\260\323(%B\034P\000\376\331\013\000\000\020\377\002Zh\000r\000\000\377\002\\h\000t\000\000rA\214\277\024@\260\323\0241z\034\377\002Lh\000x\000\000pO\214\277\f@\260\323\026\031R\034\377\002Nh\000z\000\000\377\002Ph\000|\000\000~\000\214\277\002@\260\323\020\0052\034\377\002Rh\000~\000\000|\017\214\277\f@\260\323\022\t\n\034`\000\376\331\013\000\000\002\000\200P\334/\000\000\030\000\200P\3340\000\000\031\000\200P\3341\000\0000\000\200P\3342\000\0001\000\200P\3343\000\0002\000\200P\3344\000\0003p\000\376\331\013\000\000\"\000\200P\3345\000\0004\000\200P\3346\000\0005\000\200P\3347\000\0006\000\200P\3348\000\0007\000\200P\3349\000\0008\000\200P\334:\000\0009\000\200P\334;\000\000:\000\200P\334<\000\000;xA\214\277\002@\260\323\002\0212\034\377\002^h\000\214\000\000vO\214\277\036@\260\323\004\r\n\034\000\200P\334=\000\000\026\000\200P\334>\000\000\027\000\200P\334?\000\000\024\000\200P\334\n\000\000\025\000\200P\334*\000\000\022\000\200P\334+\000\000\023\000\200P\334,\000\000\006\000\200P\334-\000\000\007\000\200P\334.\000\000\020\000\200P\334@\000\000\021\000\200P\334&\000\000\004\000\200P\334'\000\000\005\000\200P\334(\000\000\f\000\200P\334)\000\000\r\000\200P\334A\000\000\002\000\200P\334B\000\000\003\000\200P\334C\000\000\b\000\200P\334D\000\000\t\377\002L(\000\240\000\000\377\002Nh\000\242\000\000\377\002xh\000\226\000\000\377\002\024h\000\230\000\000\377\002zh\000\254\000\000\377\002|h\000\256\000\000\377\002~(\000\260\000\000\377\002\200h\000\262\000\000\377\002\202h\000\264\000\000\377\002\204h\000\266\000\000\377\002\206h\000\304\000\000\377\002\210h\000\306\000\000\220\000\376\331\013\000\000(v\201\214\277\"@\260\323\"Az\034\200\000\376\331\013\000\000\036t\217\214\277\034@\260\323$9\212\034\377\002Fh\000\232\000\000\377\002Hh\000\234\000\000\377\002Jh\000\236\000\000r\200\214\277\032@\260\323\0365r\034p\217\214\277\016@\260\323 \035j\034\000\200P\334E\000\000\032\000\200P\334F\000\000\033\000\200P\334/\000\000\034\000\200P\334G\000\000\035\000\200P\334H\000\000\036\000\200P\334I\000\000\037\000\200P\334J\000\000 \000\200P\334<\000\000!\000\200P\334\n\000\000\"\000\200P\334#\000\000#\000\200P\334$\000\000$\000\200P\334%\000\000%\000\200P\334&\000\000&\000\200P\334'\000\000'\240\000\376\331\013\000\000,\377\002\024h\000\244\000\000\377\002xh\000\246\000\000\377\002\212h\000\310\000\000|\217\214\277\016@\260\323(1:\034\377\0020h\000\250\000\000z\217\214\277\016@\260\323*a:\034\260\000\376\331\013\000\000(x\201\214\277\016@\260\323,e:\034\377\0022h\000\252\000\000v\217\214\277\016@\260\323.i:\034\300\000\376\331\013\000\000,t\201\214\277\016@\260\323(m:\034\377\002lh\000\270\000\000r\217\214\277\016@\260\323*q:\034\320\000\376\331\013\000\000(p\201\214\277\016@\260\323,u:\034\377\002nh\000\272\000\000~O\214\277\016@\260\323.-:\034\377\002ph\000\274\000\000|@\214\277\016@\260\323():\034\340\000\376\331\013\000\000\024zO\214\277\016@\260\323*%:\034\360\000\376\331\013\000\000(\377\002rh\000\276\000\000\377\002t(\000\300\000\000xA\214\277\006@\260\323\024\r:\034\377\002vh\000\302\000\000vO\214\277\006@\260\323\026!\032\034\000\001\376\331\013\000\000\016\020\001\376\331\013\000\000,tB\214\277\004@\260\323(\t\032\034rO\214\277\004@\260\323*\031\022\034 \001\376\331\013\000\000(0\001\376\331\013\000\0000pC\214\277\002@\260\323\016\005\022\034~\017\214\2774@\260\323\020\021\n\034@\001\376\331\013\000\000\006P\001\376\331\013\000\000\002\000\200P\334\n\000\000\020\000\200P\334<\000\000\021\000\200P\334\030\000\000\016\000\200P\334\031\000\000\017\000\200P\334=\000\000\f\000\200P\334>\000\000\r\000\200P\334?\000\000\024\000\200P\334@\000\000\025\000\200P\334A\000\000\022\000\200P\334B\000\000\023\000\200P\3346\000\000\030\000\200P\3347\000\000\031\000\200P\3348\000\000\026\000\200P\3349\000\000\027\377\002\024h\000\312\000\000\377\002lh\000\370\000\000\377\002nh\000\372\000\000\377\002ph\000\374\000\000zD\214\277\032@\260\323,5\322\034\377\002Xh\000\344\000\000xO\214\277\032@\260\323.9j\034\377\002Zh\000\346\000\000vC\214\277\032@\260\323(=j\034\377\002<h\000\314\000\000tO\214\277\032@\260\323*Aj\034\377\002>h\000\316\000\000rB\214\277\032@\260\3230Ej\034\377\002Fh\000\322\000\000pO\214\277\032@\260\3232Ij\034\377\002Hh\000\324\000\000~\001\214\277\006@\260\323\006Mj\034\000\200P\334:\000\000\034\000\200P\334;\000\000\035\000\200P\334C\000\000\032\000\200P\334D\000\000\033\000\200P\334E\000\000 \000\200P\334\n\000\000!\000\200P\334\036\000\000\036\000\200P\334\037\000\000\037\377\002\024(\000\320\000\000\377\002Jh\000\326\000\000\377\002Lh\000\330\000\000\377\002Nh\000\332\000\000\377\002Ph\000\334\000\000\377\002Rh\000\336\000\000\000\200P\334\n\000\000\"\000\200P\334#\000\000#\000\200P\334$\000\000$\000\200P\334%\000\000%\000\200P\334&\000\000&\000\200P\334'\000\000'\000\200P\334(\000\000(\000\200P\334)\000\000)\377\002\024(\000\340\000\000\377\002Vh\000\342\000\000\377\002\\h\000\350\000\000\377\002^h\000\352\000\000\377\002`h\000\354\000\000\377\002bh\000\356\000\000\000\200P\334\n\000\000*\000\200P\334+\000\000+\000\200P\334,\000\000,\000\200P\334-\000\000-\000\200P\334.\000\000.\000\200P\334/\000\000/\000\200P\3340\000\0000\000\200P\3341\000\0001\377\002\024(\000\360\000\000\377\002fh\000\362\000\000\377\002hh\000\364\000\000\377\002jh\000\366\000\000\377\002\002h\000\376\000\000\000\200P\334\n\000\0002\000\200P\3343\000\0003\000\200P\3344\000\0004\000\200P\3345\000\0005\000\200P\3346\000\0006\000\200P\3347\000\0007\000\200P\3348\000\0008\000\200P\334\001\000\0009\000\002l\330\013\000\000\001|\217\214\277\020@\260\323\b!\032\034`\001\376\331\013\000\000\006z\202\214\277\002@\260\323\002\035B\034p\001\376\331\013\000\000\016x\217\214\277\002@\260\323\004\031\n\034v\201\214\277\006@\260\323\006)\n\034\200\001\376\331\013\000\000\002t\217\214\277\006@\260\323\b%\032\034r\201\214\277\f@\260\323\0161\032\034\220\001\376\331\013\000\000\006p\217\214\277\f@\260\323\020-2\034~A\214\277\002@\260\323\00292\034\240\001\376\331\013\000\000\f|O\214\277\002@\260\323\0045\n\034zA\214\277\006@\260\323\006A\n\034\260\001\376\331\013\000\000\002xO\214\277\006@\260\323\b=\032\034vA\214\277\f@\260\323\fE\032\034\300\001\376\331\013\000\000\006tO\214\277\f@\260\323\016I2\034rA\214\277\002@\260\323\002M2\034\320\001\376\331\013\000\000\fpO\214\277\002@\260\323\004Q\n\034~\001\214\277\006@\260\323\006U\n\034\340\001\376\331\013\000\000\002|\017\214\277\006@\260\323\bY\032\034z\001\214\277\f@\260\323\f]\032\034\360\001\376\331\013\000\000\006x\017\214\277\f@\260\323\016a2\034v\001\214\277\002@\260\323\002e2\034t\017\214\277\002@\260\323\004i\n\034r\000\214\277\002@\260\323\006m\n\034p\017\214\277\002@\260\323\bq\n\034\000\000\200\277\002\007\004\002\001\005\002\n\002\000\375\321\002\016\001\004\237\004\006\"\002\000\b\322\002\005\021\000\000\200p\334\002\001\177\000\000\000\201\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\300\000\002\300(\000\000\000\177\300\214\277\002\003\003\277u\005\205\277\000\001\016\300\000\000\000\000\002\264\f\222\206\000\002 \002\264\203\226\203\002\002$\177\300\214\277\006\f\006\200\007\003\007\202\004\200T\334\001\000\006\004\024\200T\334\001\000\006\006$\200T\334\001\000\006\b\200\002\006~\000\034\206\276\006\377\006\200h\263\377\377\007\377\007\202\377\377\377\377\000Bp\334\000\001\177\000s\017\214\277\004\000\220\322\000\t\002\000r\017\214\277\006\000\220\322\000\r\002\000\201\b\004&q\017\214\277\b\000\220\322\000\021\002\000\203\f\f$\202\004\004$\204\020\016$\004\000\b\322\006\000\t\004\210\f\004&\004\000\b\322\004\001\t\004\220\016\004&\004\000\b\322\004\001\t\004\000\200P\334\004\000\177\004\202\000\004$\377\004\nh\000\002\000\000\377\004\fh\000\004\000\000\377\004\016h\000\006\000\000\377\004\020h\000\b\000\000\377\004\022h\000\n\000\000\377\004\024h\000\f\000\000\377\004\026h\000\016\000\000\377\004\034(\000\020\000\000\377\004\036h\000\022\000\000\377\004\204h\000\024\000\000\377\004\206h\000\026\000\000\377\004\320h\000\030\000\000\377\004\322h\000\032\000\000\377\004\324h\000\034\000\000\377\004\326h\000\036\000\000\377\004\344h\000,\000\000\377\004\346h\000.\000\000\377\004 (\0000\000\000\377\004\"h\0002\000\000\377\004$h\0004\000\000\377\004&h\0006\000\000\377\004((\000 \000\000\377\004*h\000\"\000\000\377\004,h\000$\000\000\377\004.h\000&\000\000\377\004\300h\000(\000\000\377\004\302h\000*\000\000\377\004\304h\0008\000\000\377\004\306h\000:\000\000\377\004\350h\000<\000\000\377\004\352h\000>\000\000\377\004>h\000B\000\000\377\004Lh\000D\000\000\377\004Nh\000F\000\000\377\004<h\000H\000\000\377\004:h\000J\000\000\377\0046h\000L\000\000\377\0048h\000N\000\000\377\0044(\000P\000\000\377\0042h\000R\000\000\377\0040h\000T\000\000\377\004^h\000V\000\000\377\004`h\000X\000\000\377\004bh\000Z\000\000\377\004dh\000\\\000\000\377\004fh\000^\000\000\377\004h(\000`\000\000\377\004jh\000b\000\000\377\004lh\000d\000\000\377\004nh\000f\000\000\377\004Dh\000h\000\000\377\004Fh\000j\000\000\377\004@h\000l\000\000\377\004Bh\000n\000\000\377\004H(\000p\000\000\377\004Jh\000r\000\000\377\004|h\000t\000\000\377\004~h\000v\000\000\377\004Ph\000x\000\000\377\004Rh\000z\000\000\377\004\\h\000|\000\000\377\004Zh\000~\000\000\377\004\210(\000\200\000\000\377\004\212h\000\202\000\000\377\004\214h\000\204\000\000\377\004\216h\000\206\000\000\377\004Xh\000\210\000\000\377\004\222h\000\212\000\000\377\004\224h\000\214\000\000\377\004\226h\000\216\000\000\377\004T(\000\220\000\000\377\004Vh\000\222\000\000\377\004\234h\000\224\000\000\377\004\236h\000\226\000\000\377\004ph\000\230\000\000\377\004rh\000\232\000\000\377\004zh\000\234\000\000\377\004xh\000\236\000\000p\017\214\277\000\000\032\330\002\004\000\000\177\300\214\277\000\000\212\277\000\200P\334\002\000\b\004\000\200P\334\005\000\b\005\000\200P\334\006\000\b\006\000\200P\334\007\000\b\007\000\200P\334\b\000\b\b\000\200P\334\t\000\b\t\000\200P\334\n\000\b\f\000\200P\334\013\000\b\r\000\200P\334\016\000\b\n\000\200P\334\017\000\b\013\000\200P\334B\000\b\016\000\200P\334C\000\b\017\000\000\376\331\003\000\000X\020\000\376\331\003\000\000\\\377\004\250(\000\240\000\000\377\004\252h\000\242\000\000\377\004vh\000\244\000\000\377\004th\000\246\000\000\377\004\204(\000\260\000\000\377\004\206h\000\262\000\000\377\004\200h\000\264\000\000\377\004\202h\000\266\000\000\377\004\220h\000\270\000\000\377\004\230h\000\304\000\000\377\004\232h\000\306\000\000\377\004\244h\000\310\000\000\377\004\246h\000\312\000\000\377\004\240h\000\314\000\000\377\004\242h\000\316\000\000\377\004\310h\000\250\000\000\377\004\312h\000\252\000\000\377\004\314h\000\254\000\000\377\004\316h\000\256\000\000\377\004\366h\000\272\000\000\377\004\370h\000\274\000\000\377\004\372h\000\276\000\000\377\004\374(\000\300\000\000\377\004\376h\000\302\000\000\377\004\254(\000\320\000\000\377\004\256h\000\322\000\000\377\004\364h\000\324\000\000\377\004\362h\000\326\000\000\377\004\356h\000\330\000\000\377\004\360h\000\332\000\000\377\004\354h\000\334\000\000\377\004\002h\000\336\000\000z\001\214\277\004@\261\323X\t\002\030x\017\214\277X@\260\323Z\r\022\034\000\200P\334h\000\bh\000\200P\334i\000\bi\000\200P\334j\000\bj\000\200P\334k\000\bk\000\200P\334\024\000\bl\000\200P\334\025\000\bm\000\200P\334\026\000\bn\000\200P\334\027\000\bo\000\200P\334`\000\bp\000\200P\334a\000\bq\000\200P\334r\000\br\000\200P\334s\000\bs\000\200P\334\020\000\b\006\000\200P\334\021\000\b\007\000\200P\334\022\000\b\004\000\200P\334\023\000\b\005\000\200P\334b\000\b\022\000\200P\334c\000\b\023\000\200P\334t\000\b\020\000\200P\334u\000\b\021 \000\376\331\003\000\000\024zA\214\277\b@\260\323\\\021b\0350\000\376\331\003\000\000XxO\214\277\b@\260\323^\031\"\034@\000\376\331\003\000\000\\P\000\376\331\003\000\000`vC\214\277\b@\260\323\024\025\"\034tO\214\277t@\260\323\026\035\"\034\377\004\020(\000@\000\000\000\200P\334\b\000\b\n\000\200P\334\037\000\b\013\000\200P\334&\000\b\b\000\200P\334'\000\b\t\000\200P\334\036\000\b\016\000\200P\334\035\000\b\017\000\200P\334\033\000\b\f\000\200P\334\034\000\b\r\000\200P\334\032\000\b\026\000\200P\334\031\000\b\027\000\200P\334\030\000\b\024\000\200P\334/\000\b\025\000\200P\3340\000\b\032\000\200P\3341\000\b\033\000\200P\3342\000\b\030\000\200P\3343\000\b\031\000\200P\3344\000\b\036\000\200P\3345\000\b\037\000\200P\3346\000\b\034\000\200P\3347\000\b\035\000\200P\334\"\000\b\"\000\200P\334#\000\b#\000\200P\334 \000\b \000\200P\334!\000\b!\000\200P\334$\000\b&\000\200P\334%\000\b'\000\200P\334>\000\b$\000\200P\334?\000\b%\000\200P\334(\000\b(\000\200P\334)\000\b)\000\200P\334.\000\b0\000\200P\334-\000\b1\000\200P\334D\000\b.\000\200P\334E\000\b/\000\200P\334F\000\b6\000\200P\334G\000\b7\000\200P\334,\000\b,\000\200P\334I\000\b-\000\200P\334J\000\b4\000\200P\334K\000\b5\000\200P\334*\000\b*\000\200P\334+\000\b+\000\200P\334N\000\b2\000\200P\334O\000\b3\000\200P\3348\000\b8\000\200P\3349\000\b9\000\200P\334=\000\b>\000\200P\334<\000\b?\000\200P\334T\000\bF\000\200P\334U\000\bG\000\200P\334;\000\b<\000\200P\334:\000\b=\000\200P\334d\000\bD\000\200P\334e\000\bE\000\200P\334f\000\b:\000\200P\334g\000\b;\000\200P\334B\000\bB\000\200P\334C\000\bC\000\200P\334@\000\b@\000\200P\334A\000\bA\000\200P\334H\000\bJ\000\200P\334{\000\bK\000\200P\334|\000\bH\000\200P\334}\000\bI\000\200P\334~\000\bN\000\200P\334\177\000\bO\000\200P\334L\000\bL\000\200P\334M\000\bM\000\200P\334R\000\bR\000\200P\334S\000\bS\000\200P\334P\000\bP\000\200P\334Q\000\bQ`\000\376\331\003\000\000d~\303\214\277T@\260\323X\321\322\035\000\000\200\277T@\260\323Z\325R\035\177\302\214\277T@\260\323\\\331R\035\377\004\270h\000\342\000\000T@\260\323^\335R\035\377\004\336h\000\356\000\000\177\301\214\277T@\260\323`\341R\035\377\004\272h\000\344\000\000`@\260\323b\345R\035\000\200P\334V\000\bV\000\200P\334W\000\bW\000\200P\334z\000\bT\000\200P\334y\000\bU\000\200P\334w\000\bX\000\200P\334x\000\bY\000\200P\334v\000\bZ\000\200P\334\001\000\b[\377\004\002(\000\340\000\000\377\004\274h\000\346\000\000\377\004\276h\000\350\000\000\377\004\304h\000\352\000\000\377\004\306h\000\354\000\000\000\200P\334\001\000\bh\000\200P\334\\\000\bi\000\200P\334]\000\bj\000\200P\334^\000\bk\000\200P\334_\000\bl\000\200P\334b\000\bm\000\200P\334c\000\bn\000\200P\334o\000\bo\377\004\002(\000\360\000\000\377\004\270h\000\362\000\000\377\004\356h\000\376\000\000\377\004\272h\000\364\000\000\377\004\274h\000\366\000\000\377\004\276h\000\370\000\000\377\004\304h\000\372\000\000\377\004\306h\000\374\000\000\000\200P\334\001\000\bp\000\200P\334\\\000\bq\000\200P\334]\000\br\000\200P\334^\000\bs\000\200P\334_\000\bt\000\200P\334b\000\bu\000\200P\334c\000\bv\000\200P\334w\000\bw\\\000\375\321\002\016\001\004\237\270\272\"\\\000\b\322\\\005\021\000\000\200P\334\\\000\177\001p\000\376\331\003\000\000\\\177\301\214\277\006@\260\323d\r\202\035\200\000\376\331\003\000\000`\004@\260\323f\t\032\034\000\001\006\300 \000\000\000\177\300\214\277\022@\260\323\\%\022\034\220\000\376\331\003\000\000\004\020@\260\323^!J\034\000\000\200\277\n@\260\323`\025B\034\240\000\376\331\003\000\000\020\b@\260\323b\021*\034\177\301\214\277\004@\260\323\004\035\"\034\260\000\376\331\003\000\000\b\004@\260\323\006\031\022\034\177\301\214\277\f@\260\323\020-\022\034\300\000\376\331\003\000\000\004~\317\214\277\f@\260\323\022)2\034\177\301\214\277\b@\260\323\b52\034\320\000\376\331\003\000\000\f\b@\260\323\n1\"\034\177\301\214\277\004@\260\323\004=\"\034\340\000\376\331\003\000\000\b\004@\260\323\0069\022\034\177\301\214\277\f@\260\323\fE\022\034\360\000\376\331\003\000\000\004\f@\260\323\016A2\034\177\301\214\277\b@\260\323\bM2\034\000\001\376\331\003\000\000\f\b@\260\323\nI\"\034\177\301\214\277\004@\260\323\004Q\"\034\000\000\200\277\b@\260\323\006a\022\034\020\001\376\331\003\000\000\004\177\301\214\277\b@\260\323\f]\"\034}\317\214\277\020@\260\323\016m\"\034 \001\376\331\003\000\000\b0\001\376\331\003\000\000\f{\302\214\277\004@\260\323\004YB\034y\317\214\277\024@\260\323\006i\022\034@\001\376\331\003\000\000\004P\001\376\331\003\000\000\020w\303\214\277\b@\260\323\bUR\034u\317\214\277\b@\260\323\ne\"\034s\302\214\277\b@\260\323\fq\"\034q\317\214\277\b@\260\323\016}\"\034\177\201\214\277\004@\260\323\004\215\"\034`\001\376\331\003\000\000\b}\217\214\277\004@\260\323\006y\022\034{\201\214\277\f@\260\323\020\211\022\034p\001\376\331\003\000\000\004y\217\214\277\f@\260\323\022u2\034w\201\214\277\b@\260\323\b\2052\034\200\001\376\331\003\000\000\fu\217\214\277\b@\260\323\n\201\"\034s\201\214\277\004@\260\323\004\225\"\034\220\001\376\331\003\000\000\bq\217\214\277\004@\260\323\006\221\022\034\177A\214\277\f@\260\323\f\235\022\034\240\001\376\331\003\000\000\004}O\214\277\f@\260\323\016\2312\034{A\214\277\b@\260\323\b\2452\034\260\001\376\331\003\000\000\fyO\214\277\b@\260\323\n\241\"\034wA\214\277\004@\260\323\004\255\"\034\300\001\376\331\003\000\000\buO\214\277\004@\260\323\006\251\022\034sA\214\277\004@\260\323\f\261\022\034qO\214\277\f@\260\323\016\265\022\034\320\001\376\331\003\000\000\004\177\001\214\277\b@\260\323\b\3212\034\340\001\376\331\003\000\000\f}\017\214\277\020@\260\323\n\325\"\034\360\001\376\331\003\000\000\b{\002\214\277\004@\260\323\004\331B\034y\017\214\277\004@\260\323\006\335\022\034w\001\214\277\004@\260\323\f\341\022\034u\017\214\277\004@\260\323\016\345\022\034s\000\214\277\004@\260\323\b\351\022\034q\017\214\277\004@\260\323\n\355\022\034\000\000\200\277\004\013\006\002p\017\214\277\001\007\002\004\000\000\032\330\002\001\000\000\177\300\214\277\000\000\212\277\000\000l\330\002\000\000\001\003\000\214\322\301\000\001\000\003\000\215\322\301\006\002\000\377\002\b~\200\000\000\000\004\000\000\322\003\005\021\004\177\300\214\277\001\003\004\n\000\000~\330\004\002\000\002\277\006\b&\260\b\230}\177\300\214\277\001\003\004v\001\000\000\321\200 \251\001\001\000\376\321\001\007\n\002\000\000~\330\001\002\000\001\270\b\230}\177\300\214\277\002\003\002\002\002\000\000\321\200\020\251\001\002\000\376\321\002\007\n\002\000\000~\330\002\001\000\002\274\b\230}\177\300\214\277\001\005\002\002\002\000\000\321\200\b\251\001\002\000\376\321\002\007\n\002\000\000~\330\002\001\000\002\276\b\230}\177\300\214\277\001\005\002\002\002\000\000\321\200\004\251\001\002\000\376\321\002\007\n\002\000\000~\330\002\001\000\002\277\b\232}\177\300\214\277\001\005\004\002\200\006\0028\202\002\002$\000\000~\330\001\002\000\003\277\000\002&\200\002\224}j \200\276\006\000\210\277\206\000\002 \202\002\002$\177\300\214\277\002\007\004\002\000\002\032\330\001\002\000\000~\000\376\207\000\000\312\320\200\000\002\000\177\300\214\277\000\000\212\277\000 \206\276\007\000\210\277\200\002\002~\000\002\354\330\001\000\000\002\177\300\214\277\003\005\004\002\b\002\032\330\001\002\000\000~\006\376\207\211\000\002$\177\300\214\277\000\000\212\2770\200\\\334\001\000\n\002 \200\\\334\001\000\n|P\200\\\334\001\000\n\n@\200\\\334\001\000\n4\220\200\\\334\001\000\n\030p\200\\\334\001\000\n\020`\200\\\334\001\000\n,\320\200\\\334\001\000\n\024`\201\\\334\001\000\nh\200\002\000~P\000\376\331\000\000\000$@\000\376\331\000\000\0000\260\000\376\331\000\000\000 \002\224\203\226\002\224\002\222\177\302\214\277P@|\334\000$\177\000p\000\376\331\000\000\000&\177\301\214\277@@|\334\000 \177\000\004\002\002\200\005\003\003\202\000\200\\\334\001\000\n<\177\300\214\277p@|\334\000&\177\000\220\000\376\331\000\000\000(\000\000\376\331\000\000\0008 \000\376\331\000\000\000@`\001\376\331\000\000\000d\200\201\\\334\001\000\n`\200\001\376\331\000\000\000\\\240\201\\\334\001\000\nX\300\001\376\331\000\000\000L\340\201\\\334\001\000\nH\000\001\376\331\000\000\000p\240\001\376\331\000\000\000T\300\201\\\334\001\000\nP\340\001\376\331\000\000\000DpO\214\277\240@|\334\000\002\177\000\020\200\\\334\001\000\n\004pO\214\277\300@|\334\000\n\177\000\260\200\\\334\001\000\n\fpO\214\2770@|\334\000\030\177\000\020\000\376\331\000\000\000\032pO\214\277`@|\334\000\020\177\000\177\017\214\277\020@|\334\000\024\177\000\177\300\214\277\320@|\334\000\032\177\000\240\200\\\334\001\000\n\034w\017\214\277\220@|\334\000\004\177\0000\000\376\331\000\000\000\006v\017\214\277 @|\334\000\f\177\000\177\300\214\277\002@\261\323\006\005\002\030\260@|\334\000\006\177\000\002@\260\323\032\t\n\034\320\000\376\331\000\000\000\004\002@\260\323$\025\n\034\360\000\376\331\000\000\000\b\002@\260\323&!\n\034\300\000\376\331\000\000\000\020\002@\260\323(1\n\034\240\000\376\331\000\000\000\030\002@\260\323 \031\n\034\177\303\214\277\000@|\334\000\004\177\000\002@\260\323\004)\n\034\360\200\\\334\001\000\n\004\177\302\214\277\340A|\334\000\b\177\000\200\200\\\334\001\000\n$\300\200\\\334\001\000\n\024\340\200\\\334\001\000\n\f\200@|\334\000(\177\000`\000\376\331\000\000\000(\200\000\376\331\000\000\000 u\017\214\277\360A|\334\000\004\177\000\002@\260\323\b\t\n\034\020\201\\\334\001\000\n\004\020\001\376\331\000\000\000\b\177\300\214\277\300A|\334\000\b\177\000q\017\214\277\320A|\334\000\004\177\000\002@\260\323\b\t\n\0340\201\\\334\001\000\n\0040\001\376\331\000\000\000\b\177\300\214\277\240A|\334\000\b\177\000q\017\214\277\260A|\334\000\004\177\000\002@\260\323\b\t\n\034P\201\\\334\001\000\n\004P\001\376\331\000\000\000\b\177\300\214\277\200A|\334\000\b\177\000q\017\214\277\220A|\334\000\004\177\000\002@\260\323\b\t\n\034p\201\\\334\001\000\n\004p\001\376\331\000\000\000\b\177\300\214\277`A|\334\000\b\177\000q\017\214\277pA|\334\000\004\177\000\002@\260\323\b\t\n\034\220\201\\\334\001\000\n\004\220\001\376\331\000\000\000\b\177\300\214\277@A|\334\000\b\177\000q\017\214\277PA|\334\000\004\177\000\002@\260\323\b\t\n\034\260\201\\\334\001\000\n\004\260\001\376\331\000\000\000\b\177\300\214\277 A|\334\000\b\177\000q\017\214\2770A|\334\000\004\177\000\002@\260\323\b\t\n\034\320\201\\\334\001\000\n\004\320\001\376\331\000\000\000\b\177\300\214\277\000A|\334\000\b\177\000q\017\214\277\020A|\334\000\004\177\000\002@\260\323\b\t\n\034\360\201\\\334\001\000\n\004\360\001\376\331\000\000\000\b\177\300\214\277\340@|\334\000\b\177\000q\017\214\277\360@|\334\000\004\177\000\002@\260\323\b\t\n\034\000\201\\\334\001\000\n\004\002@\260\323@\371\n\034\340\000\376\331\000\000\000\b\002@\260\3238y\n\034@\201\\\334\001\000\nz\002@\260\3230i\n\034\000\000\200\277\002@\260\323(Y\n\034\000\000\200\277\002@\260\323 I\n\034\000\000\200\277\002@\260\323\0309\n\034\000\000\200\277\002@\260\323\020)\n\034\177\300\214\277\002@\260\323\b\031\n\034q\017\214\277\b@\260\323p\t\n\034 \201\\\334\001\000\n\002 \001\376\331\000\000\000np\000\214\277\002@\260\323n\005\"\034@\001\376\331\000\000\000l\177\300\214\277\002@\260\323l\365\n\034\240@\\\334\000\000\177x\260@\\\334\000\000\177t\002@\260\323d\321\n\034\000\000\200\277\002@\260\323\\\301\n\034\000\000\200\277\002@\260\323T\261\n\034\000\000\200\277\002@\260\323L\241\n\034\000\000\200\277\002@\260\323D\221\n\034p\017\214\277\002@\260\323v\365\n\034\220@\\\334\000\000\177x\320@\\\334\000\000\177tp\017\214\277\002@\260\323v\365\n\034P@\\\334\000\000\177v\300@\\\334\000\000\177tp\017\214\277\002@\260\323x\355\n\034`@\\\334\000\000\177vp@\\\334\000\000\177tp\017\214\277\002@\260\323v\361\n\0340@\\\334\000\000\177x\200@\\\334\000\000\177tp\017\214\277\002@\260\323v\365\n\034 @\\\334\000\000\177x@@\\\334\000\000\177tp\017\214\277\002@\260\323v\365\n\034\000@\\\334\000\000\177t\020@\\\334\000\000\177xp\017\214\277\002@\260\323v\365\n\034\340A\\\334\000\000\177t\360A\\\334\000\000\177xp\017\214\277\002@\260\323v\365\n\034\300A\\\334\000\000\177t\320A\\\334\000\000\177xp\017\214\277\002@\260\323v\365\n\034\240A\\\334\000\000\177t\260A\\\334\000\000\177xp\017\214\277\002@\260\323v\365\n\034\200A\\\334\000\000\177t\220A\\\334\000\000\177xp\017\214\277\002@\260\323v\365\n\034`A\\\334\000\000\177tpA\\\334\000\000\177xp\017\214\277\002@\260\323v\365\n\034@A\\\334\000\000\177tPA\\\334\000\000\177xp\017\214\277\002@\260\323v\365\n\034 A\\\334\000\000\177t0A\\\334\000\000\177xp\017\214\277\002@\260\323v\365\n\034\000A\\\334\000\000\177t\020A\\\334\000\000\177xp\017\214\277\002@\260\323v\365\n\034\340@\\\334\000\000\177t\360@\\\334\000\000\177x\b\002l\330\000\000\000\000p\017\214\277\002@\260\323v\365\n\034\000\000\200\277\002@\260\323B\375\n\034\000\000\200\277\002@\260\323:}\n\034\000\000\200\277\002@\260\3232m\n\034\000\000\200\277\002@\260\323*]\n\034\000\000\200\277\002@\260\323\"M\n\034\000\000\200\277\002@\260\323\032=\n\034\000\000\200\277\002@\260\323\022-\n\034\000\000\200\277\002@\260\323\n\035\n\034\000\000\200\277\002@\260\323r\r\n\034\000\000\200\277\002@\260\323p\t\n\034\000\000\200\277\002@\260\323n\371\n\034\000\000\200\277\002@\260\323f\325\n\034\000\000\200\277\002@\260\323^\305\n\034\000\000\200\277\002@\260\323V\265\n\034\000\000\200\277\002@\260\323N\245\n\034\000\000\200\277\002@\260\323F\225\n\034\000\000\200\277\002\007\002\002\006\000A\320\200\002\002\000j \204\276\006\000\210\277\000BP\334\000\000\177\001\006p\004~p\017\214\277\000\200t\334\001\002\002\000~\004\376\207\000 \204\276\005\000\210\277\177\300\214\277\000O\000~\200\002\002~\020\200p\334\001\000\002\000\000\000\201\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\000\000\200\277\006\000\000\000\000\000\000\000\210\025\000\000\000\000\000\000\013\000\000\000\000\000\000\000\030\000\000\000\000\000\000\000\005\000\000\000\000\000\000\0000\030\000\000\000\000\000\000\n\000\000\000\000\000\000\000\361\002\000\000\000\000\000\000\365\376\377o\000\000\000\000 \027\000\000\000\000\000\000\004\000\000\000\000\000\000\000\240\027\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000AMD clang version 22.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.2.0 26014 7b800a19466229b8479a78de19143dc33c3ab9b5)\000Linker: AMD LLD 22.0.0 (/longer_pathname_so_that_rpms_can_support_packaging_the_debug_info_for_all_os_profiles/src/llvm-project/llvm 7b800a19466229b8479a78de19143dc33c3ab9b5)\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\361\377\200\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000@\000\000\000\000\000\361\377\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\177\000\000\000\000\000\361\377\020\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\303\000\000\000\000\000\361\377\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\013\001\000\000\000\000\361\377\204\002\000\000\000\000\000\000\000\000\000\000\000\000\000\000R\001\000\000\000\000\361\377\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\221\001\000\000\000\000\361\377\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\331\001\000\000\000\000\361\377\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000#\002\000\000\000\000\361\377\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000g\002\000\000\000\000\361\377\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\257\002\000\000\000\000\361\377K\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\360\002\000\000\000\000\361\377\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\0001\003\000\000\000\000\361\377\n\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000w\003\000\000\000\000\361\377\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\301\003\000\000\000\000\361\377\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\n\004\000\000\000\000\361\377\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000K\004\000\000\000\000\361\377\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\225\004\000\000\000\000\361\377\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\341\004\000\000\000\000\361\377\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000'\005\000\000\000\000\361\377\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000q\005\000\000\000\000\361\377\f\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\262\005\000\000\000\000\361\377\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\363\005\000\000\000\000\361\377\016\000\000\000\000\000\000\000\000\000\000\000\000\000\000\0009\006\000\000\000\000\361\377\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\203\006\000\000\000\000\361\377\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\314\006\000\000\000\000\361\377\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\r\007\000\000\000\000\361\377\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000W\007\000\000\000\000\361\377\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\243\007\000\000\000\000\361\377\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\351\007\000\000\000\000\361\377\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\0003\b\000\000\000\000\361\377\200\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000r\b\000\000\000\000\361\377\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\261\b\000\000\000\000\361\377\030\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\365\b\000\000\000\000\361\377\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000=\t\000\000\000\000\361\377\204\002\000\000\000\000\000\000\000\000\000\000\000\000\000\000\204\t\000\000\000\000\361\377\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\303\t\000\000\000\000\361\377\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\013\n\000\000\000\000\361\377\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000U\n\000\000\000\000\361\377\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\231\n\000\000\000\000\361\377\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\341\n\000\000\000\000\361\377K\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\"\013\000\000\000\000\361\377\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000c\013\000\000\000\000\361\377\n\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\251\013\000\000\000\000\361\377\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\363\013\000\000\000\000\361\377\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000<\f\000\000\000\000\361\377\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000}\f\000\000\000\000\361\377\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\307\f\000\000\000\000\361\377\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\023\r\000\000\000\000\361\377\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000Y\r\000\000\000\000\361\377\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\243\r\000\000\000\000\361\377\200\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\360\r\000\000\000\000\361\377\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000=\016\000\000\000\000\361\377\r\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\217\016\000\000\000\000\361\377\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\345\016\000\000\000\000\361\377\b\002\000\000\000\000\000\000\000\000\000\000\000\000\000\000:\017\000\000\000\000\361\377\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\207\017\000\000\000\000\361\377\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\335\017\000\000\000\000\361\377\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\0005\020\000\000\000\000\361\377\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\207\020\000\000\000\000\361\377\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\335\020\000\000\000\000\361\377\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\361\020\000\000\000\000\361\377\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\005\021\000\000\000\000\361\377\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\t\024\000\000\000\002\b\000\000\223\000\000\000\000\000\000\000\000\000\000\000\000\000\000\031\021\000\000\022\003\007\000\000.\000\000\000\000\000\000\224\f\000\000\000\000\000\000O\021\000\000\021\003\006\000\300\034\000\000\000\000\000\000 \000\000\000\000\000\000\000U\021\000\000\021\003\006\000@\033\000\000\000\000\000\000@\000\000\000\000\000\000\000\216\021\000\000\022\003\007\000\000;\000\000\000\000\000\000t\r\000\000\000\000\000\000\306\021\000\000\021\003\006\000\200\033\000\000\000\000\000\000@\000\000\000\000\000\000\000\001\022\000\000\022\003\007\000\000I\000\000\000\000\000\000\b\003\000\000\000\000\000\0009\022\000\000\021\003\006\000\300\033\000\000\000\000\000\000@\000\000\000\000\000\000\000t\022\000\000\022\003\007\000\000M\000\000\000\000\000\000\200\r\000\000\000\000\000\000\252\022\000\000\021\003\006\000\340\034\000\000\000\000\000\000@\000\000\000\000\000\000\000\260\022\000\000\021\003\006\000\000\034\000\000\000\000\000\000@\000\000\000\000\000\000\000\351\022\000\000\022\003\007\000\000[\000\000\000\000\000\000\240\r\000\000\000\000\000\000!\023\000\000\021\003\006\000@\034\000\000\000\000\000\000@\000\000\000\000\000\000\000\\\023\000\000\022\003\007\000\000i\000\000\000\000\000\000\354\025\000\000\000\000\000\000\240\023\000\000\021\003\006\000\200\034\000\000\000\000\000\000@\000\000\000\000\000\000\000\347\023\000\000\021\003\006\000 \035\000\000\000\000\000\000\020\000\000\000\000\000\000\000\355\023\000\000\021\000\n\000p\243\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000.note\000.dynsym\000.gnu.hash\000.hash\000.dynstr\000.rodata\000.text\000.dynamic\000.relro_padding\000.bss\000.AMDGPU.csdata\000.AMDGPU.gpr_maximums\000.comment\000.symtab\000.shstrtab\000.strtab\000\000_Z23tqm_quantize_kernel_tq3PKfS0_P16block_tq3_mi300xi.num_vgpr\000_Z23tqm_quantize_kernel_tq3PKfS0_P16block_tq3_mi300xi.num_agpr\000_Z23tqm_quantize_kernel_tq3PKfS0_P16block_tq3_mi300xi.numbered_sgpr\000_Z23tqm_quantize_kernel_tq3PKfS0_P16block_tq3_mi300xi.num_named_barrier\000_Z23tqm_quantize_kernel_tq3PKfS0_P16block_tq3_mi300xi.private_seg_size\000_Z23tqm_quantize_kernel_tq3PKfS0_P16block_tq3_mi300xi.uses_vcc\000_Z23tqm_quantize_kernel_tq3PKfS0_P16block_tq3_mi300xi.uses_flat_scratch\000_Z23tqm_quantize_kernel_tq3PKfS0_P16block_tq3_mi300xi.has_dyn_sized_stack\000_Z23tqm_quantize_kernel_tq3PKfS0_P16block_tq3_mi300xi.has_recursion\000_Z23tqm_quantize_kernel_tq3PKfS0_P16block_tq3_mi300xi.has_indirect_call\000_Z25tqm_dequantize_kernel_tq3PK16block_tq3_mi300xPKfPfi.num_vgpr\000_Z25tqm_dequantize_kernel_tq3PK16block_tq3_mi300xPKfPfi.num_agpr\000_Z25tqm_dequantize_kernel_tq3PK16block_tq3_mi300xPKfPfi.numbered_sgpr\000_Z25tqm_dequantize_kernel_tq3PK16block_tq3_mi300xPKfPfi.num_named_barrier\000_Z25tqm_dequantize_kernel_tq3PK16block_tq3_mi300xPKfPfi.private_seg_size\000_Z25tqm_dequantize_kernel_tq3PK16block_tq3_mi300xPKfPfi.uses_vcc\000_Z25tqm_dequantize_kernel_tq3PK16block_tq3_mi300xPKfPfi.uses_flat_scratch\000_Z25tqm_dequantize_kernel_tq3PK16block_tq3_mi300xPKfPfi.has_dyn_sized_stack\000_Z25tqm_dequantize_kernel_tq3PK16block_tq3_mi300xPKfPfi.has_recursion\000_Z25tqm_dequantize_kernel_tq3PK16block_tq3_mi300xPKfPfi.has_indirect_call\000_Z24tqm_fused_dot_kernel_tq3PKfPK16block_tq3_mi300xPfii.num_vgpr\000_Z24tqm_fused_dot_kernel_tq3PKfPK16block_tq3_mi300xPfii.num_agpr\000_Z24tqm_fused_dot_kernel_tq3PKfPK16block_tq3_mi300xPfii.numbered_sgpr\000_Z24tqm_fused_dot_kernel_tq3PKfPK16block_tq3_mi300xPfii.num_named_barrier\000_Z24tqm_fused_dot_kernel_tq3PKfPK16block_tq3_mi300xPfii.private_seg_size\000_Z24tqm_fused_dot_kernel_tq3PKfPK16block_tq3_mi300xPfii.uses_vcc\000_Z24tqm_fused_dot_kernel_tq3PKfPK16block_tq3_mi300xPfii.uses_flat_scratch\000_Z24tqm_fused_dot_kernel_tq3PKfPK16block_tq3_mi300xPfii.has_dyn_sized_stack\000_Z24tqm_fused_dot_kernel_tq3PKfPK16block_tq3_mi300xPfii.has_recursion\000_Z24tqm_fused_dot_kernel_tq3PKfPK16block_tq3_mi300xPfii.has_indirect_call\000_Z23tqm_quantize_kernel_tq4PKfS0_P16block_tq4_mi300xi.num_vgpr\000_Z23tqm_quantize_kernel_tq4PKfS0_P16block_tq4_mi300xi.num_agpr\000_Z23tqm_quantize_kernel_tq4PKfS0_P16block_tq4_mi300xi.numbered_sgpr\000_Z23tqm_quantize_kernel_tq4PKfS0_P16block_tq4_mi300xi.num_named_barrier\000_Z23tqm_quantize_kernel_tq4PKfS0_P16block_tq4_mi300xi.private_seg_size\000_Z23tqm_quantize_kernel_tq4PKfS0_P16block_tq4_mi300xi.uses_vcc\000_Z23tqm_quantize_kernel_tq4PKfS0_P16block_tq4_mi300xi.uses_flat_scratch\000_Z23tqm_quantize_kernel_tq4PKfS0_P16block_tq4_mi300xi.has_dyn_sized_stack\000_Z23tqm_quantize_kernel_tq4PKfS0_P16block_tq4_mi300xi.has_recursion\000_Z23tqm_quantize_kernel_tq4PKfS0_P16block_tq4_mi300xi.has_indirect_call\000_Z25tqm_dequantize_kernel_tq4PK16block_tq4_mi300xPKfPfi.num_vgpr\000_Z25tqm_dequantize_kernel_tq4PK16block_tq4_mi300xPKfPfi.num_agpr\000_Z25tqm_dequantize_kernel_tq4PK16block_tq4_mi300xPKfPfi.numbered_sgpr\000_Z25tqm_dequantize_kernel_tq4PK16block_tq4_mi300xPKfPfi.num_named_barrier\000_Z25tqm_dequantize_kernel_tq4PK16block_tq4_mi300xPKfPfi.private_seg_size\000_Z25tqm_dequantize_kernel_tq4PK16block_tq4_mi300xPKfPfi.uses_vcc\000_Z25tqm_dequantize_kernel_tq4PK16block_tq4_mi300xPKfPfi.uses_flat_scratch\000_Z25tqm_dequantize_kernel_tq4PK16block_tq4_mi300xPKfPfi.has_dyn_sized_stack\000_Z25tqm_dequantize_kernel_tq4PK16block_tq4_mi300xPKfPfi.has_recursion\000_Z25tqm_dequantize_kernel_tq4PK16block_tq4_mi300xPKfPfi.has_indirect_call\000_Z14tqm_qjl_kernelPKfPK16block_tq3_mi300xS0_S0_P16block_qjl_mi300xi.num_vgpr\000_Z14tqm_qjl_kernelPKfPK16block_tq3_mi300xS0_S0_P16block_qjl_mi300xi.num_agpr\000_Z14tqm_qjl_kernelPKfPK16block_tq3_mi300xS0_S0_P16block_qjl_mi300xi.numbered_sgpr\000_Z14tqm_qjl_kernelPKfPK16block_tq3_mi300xS0_S0_P16block_qjl_mi300xi.num_named_barrier\000_Z14tqm_qjl_kernelPKfPK16block_tq3_mi300xS0_S0_P16block_qjl_mi300xi.private_seg_size\000_Z14tqm_qjl_kernelPKfPK16block_tq3_mi300xS0_S0_P16block_qjl_mi300xi.uses_vcc\000_Z14tqm_qjl_kernelPKfPK16block_tq3_mi300xS0_S0_P16block_qjl_mi300xi.uses_flat_scratch\000_Z14tqm_qjl_kernelPKfPK16block_tq3_mi300xS0_S0_P16block_qjl_mi300xi.has_dyn_sized_stack\000_Z14tqm_qjl_kernelPKfPK16block_tq3_mi300xS0_S0_P16block_qjl_mi300xi.has_recursion\000_Z14tqm_qjl_kernelPKfPK16block_tq3_mi300xS0_S0_P16block_qjl_mi300xi.has_indirect_call\000amdgpu.max_num_vgpr\000amdgpu.max_num_agpr\000amdgpu.max_num_sgpr\000_Z23tqm_quantize_kernel_tq3PKfS0_P16block_tq3_mi300xi\000d_cb3\000_Z23tqm_quantize_kernel_tq3PKfS0_P16block_tq3_mi300xi.kd\000_Z25tqm_dequantize_kernel_tq3PK16block_tq3_mi300xPKfPfi\000_Z25tqm_dequantize_kernel_tq3PK16block_tq3_mi300xPKfPfi.kd\000_Z24tqm_fused_dot_kernel_tq3PKfPK16block_tq3_mi300xPfii\000_Z24tqm_fused_dot_kernel_tq3PKfPK16block_tq3_mi300xPfii.kd\000_Z23tqm_quantize_kernel_tq4PKfS0_P16block_tq4_mi300xi\000d_cb4\000_Z23tqm_quantize_kernel_tq4PKfS0_P16block_tq4_mi300xi.kd\000_Z25tqm_dequantize_kernel_tq4PK16block_tq4_mi300xPKfPfi\000_Z25tqm_dequantize_kernel_tq4PK16block_tq4_mi300xPKfPfi.kd\000_Z14tqm_qjl_kernelPKfPK16block_tq3_mi300xS0_S0_P16block_qjl_mi300xi\000_Z14tqm_qjl_kernelPKfPK16block_tq3_mi300xS0_S0_P16block_qjl_mi300xi.kd\000d_cb2\000__hip_cuid_81ebb66926e4b9a5\000_DYNAMIC\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\007\000\000\000\002\000\000\000\000\000\000\0008\002\000\000\000\000\000\0008\002\000\000\000\000\000\000L\023\000\000\000\000\000\000\000\000\000\000\000\000\000\000\004\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\007\000\000\000\013\000\000\000\002\000\000\000\000\000\000\000\210\025\000\000\000\000\000\000\210\025\000\000\000\000\000\000\230\001\000\000\000\000\000\000\005\000\000\000\001\000\000\000\b\000\000\000\000\000\000\000\030\000\000\000\000\000\000\000\017\000\000\000\366\377\377o\002\000\000\000\000\000\000\000 \027\000\000\000\000\000\000 \027\000\000\000\000\000\000\200\000\000\000\000\000\000\000\002\000\000\000\000\000\000\000\b\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\031\000\000\000\005\000\000\000\002\000\000\000\000\000\000\000\240\027\000\000\000\000\000\000\240\027\000\000\000\000\000\000\220\000\000\000\000\000\000\000\002\000\000\000\000\000\000\000\004\000\000\000\000\000\000\000\004\000\000\000\000\000\000\000\037\000\000\000\003\000\000\000\002\000\000\000\000\000\000\0000\030\000\000\000\000\000\0000\030\000\000\000\000\000\000\361\002\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000'\000\000\000\001\000\000\000\002\000\000\000\000\000\000\000@\033\000\000\000\000\000\000@\033\000\000\000\000\000\000\360\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000@\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000/\000\000\000\001\000\000\000\006\000\000\000\000\000\000\000\000.\000\000\000\000\000\000\000\036\000\000\000\000\000\000\000U\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\0005\000\000\000\006\000\000\000\003\000\000\000\000\000\000\000\000\223\000\000\000\000\000\000\000s\000\000\000\000\000\000p\000\000\000\000\000\000\000\005\000\000\000\000\000\000\000\b\000\000\000\000\000\000\000\020\000\000\000\000\000\000\000>\000\000\000\b\000\000\000\003\000\000\000\000\000\000\000p\223\000\000\000\000\000\000ps\000\000\000\000\000\000\220\f\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000M\000\000\000\b\000\000\000\003\000\000\000\000\000\000\000p\243\000\000\000\000\000\000ps\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000R\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000ps\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000a\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000ps\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000v\000\000\000\001\000\000\0000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000ps\000\000\000\000\000\0009\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\177\000\000\000\002\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\260t\000\000\000\000\000\000\230\007\000\000\000\000\000\000\020\000\000\000A\000\000\000\b\000\000\000\000\000\000\000\030\000\000\000\000\000\000\000\207\000\000\000\003\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000H|\000\000\000\000\000\000\231\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\221\000\000\000\003\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\341|\000\000\000\000\000\000\022\024\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000"
	.size	.L__unnamed_10, 42296

	.type	__hip_fatbin_wrapper,@object    # @__hip_fatbin_wrapper
	.section	.hipFatBinSegment,"aw",@progbits
	.p2align	3, 0x0
__hip_fatbin_wrapper:
	.long	1212764230                      # 0x48495046
	.long	1                               # 0x1
	.quad	.L__unnamed_10
	.quad	0
	.size	__hip_fatbin_wrapper, 24

	.type	__hip_gpubin_handle_81ebb66926e4b9a5,@object # @__hip_gpubin_handle_81ebb66926e4b9a5
	.local	__hip_gpubin_handle_81ebb66926e4b9a5
	.comm	__hip_gpubin_handle_81ebb66926e4b9a5,8,8
	.section	.init_array,"aw",@init_array
	.p2align	3, 0x0
	.quad	__hip_module_ctor
	.type	__hip_cuid_81ebb66926e4b9a5,@object # @__hip_cuid_81ebb66926e4b9a5
	.bss
	.globl	__hip_cuid_81ebb66926e4b9a5
__hip_cuid_81ebb66926e4b9a5:
	.byte	0                               # 0x0
	.size	__hip_cuid_81ebb66926e4b9a5, 1

	.ident	"AMD clang version 22.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.2.0 26014 7b800a19466229b8479a78de19143dc33c3ab9b5)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym _Z38__device_stub__tqm_quantize_kernel_tq3PKfS0_P16block_tq3_mi300xi
	.addrsig_sym _Z40__device_stub__tqm_dequantize_kernel_tq3PK16block_tq3_mi300xPKfPfi
	.addrsig_sym _Z39__device_stub__tqm_fused_dot_kernel_tq3PKfPK16block_tq3_mi300xPfii
	.addrsig_sym _Z38__device_stub__tqm_quantize_kernel_tq4PKfS0_P16block_tq4_mi300xi
	.addrsig_sym _Z40__device_stub__tqm_dequantize_kernel_tq4PK16block_tq4_mi300xPKfPfi
	.addrsig_sym _Z29__device_stub__tqm_qjl_kernelPKfPK16block_tq3_mi300xS0_S0_P16block_qjl_mi300xi
	.addrsig_sym __hip_module_ctor
	.addrsig_sym __hip_module_dtor
	.addrsig_sym d_cb3
	.addrsig_sym d_cb4
	.addrsig_sym d_cb2
	.addrsig_sym _Z23tqm_quantize_kernel_tq3PKfS0_P16block_tq3_mi300xi
	.addrsig_sym _Z25tqm_dequantize_kernel_tq3PK16block_tq3_mi300xPKfPfi
	.addrsig_sym _Z24tqm_fused_dot_kernel_tq3PKfPK16block_tq3_mi300xPfii
	.addrsig_sym _Z23tqm_quantize_kernel_tq4PKfS0_P16block_tq4_mi300xi
	.addrsig_sym _Z25tqm_dequantize_kernel_tq4PK16block_tq4_mi300xPKfPfi
	.addrsig_sym _Z14tqm_qjl_kernelPKfPK16block_tq3_mi300xS0_S0_P16block_qjl_mi300xi
	.addrsig_sym _ZL19TQ3_CODEBOOK_MI300X
	.addrsig_sym _ZL19TQ4_CODEBOOK_MI300X
	.addrsig_sym _ZL19TQ2_CODEBOOK_MI300X
	.addrsig_sym .L__unnamed_10
	.addrsig_sym __hip_fatbin_wrapper
	.addrsig_sym __hip_cuid_81ebb66926e4b9a5
