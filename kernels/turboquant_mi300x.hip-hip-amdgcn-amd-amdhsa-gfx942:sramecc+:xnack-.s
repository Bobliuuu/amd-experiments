	.amdgcn_target "amdgcn-amd-amdhsa--gfx942:sramecc+:xnack-"
	.amdhsa_code_object_version 6
	.text
	.protected	_Z23tqm_quantize_kernel_tq3PKfS0_P16block_tq3_mi300xi ; -- Begin function _Z23tqm_quantize_kernel_tq3PKfS0_P16block_tq3_mi300xi
	.globl	_Z23tqm_quantize_kernel_tq3PKfS0_P16block_tq3_mi300xi
	.p2align	8
	.type	_Z23tqm_quantize_kernel_tq3PKfS0_P16block_tq3_mi300xi,@function
_Z23tqm_quantize_kernel_tq3PKfS0_P16block_tq3_mi300xi: ; @_Z23tqm_quantize_kernel_tq3PKfS0_P16block_tq3_mi300xi
; %bb.0:
	s_load_dword s3, s[0:1], 0x18
	s_waitcnt lgkmcnt(0)
	s_cmp_ge_i32 s2, s3
	s_cbranch_scc1 .LBB0_20
; %bb.1:
	s_load_dwordx2 s[4:5], s[0:1], 0x0
	s_load_dwordx2 s[8:9], s[0:1], 0x10
	v_lshl_add_u32 v2, s2, 7, v0
	v_ashrrev_i32_e32 v3, 31, v2
	s_waitcnt lgkmcnt(0)
	v_lshl_add_u64 v[2:3], v[2:3], 2, s[4:5]
	global_load_dword v1, v[2:3], off
	v_lshlrev_b32_e32 v2, 2, v0
	v_mov_b32_e32 v3, 0x80
	s_waitcnt vmcnt(0)
	ds_write_b32 v2, v1
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b32 v1, v2
	v_mbcnt_lo_u32_b32 v2, -1, 0
	v_mbcnt_hi_u32_b32 v2, -1, v2
	v_lshl_or_b32 v3, v2, 2, v3
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v4, v1, v1
	ds_bpermute_b32 v3, v3, v4
	v_and_b32_e32 v4, 63, v2
	v_cmp_gt_u32_e32 vcc, 48, v4
	s_waitcnt lgkmcnt(0)
	v_fmac_f32_e32 v3, v1, v1
	v_cndmask_b32_e64 v5, 0, 16, vcc
	v_add_lshl_u32 v5, v5, v2, 2
	ds_bpermute_b32 v1, v5, v3
	v_cmp_gt_u32_e32 vcc, 56, v4
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v1, v3, v1
	v_cndmask_b32_e64 v5, 0, 8, vcc
	v_add_lshl_u32 v5, v5, v2, 2
	ds_bpermute_b32 v3, v5, v1
	v_cmp_gt_u32_e32 vcc, 60, v4
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v1, v1, v3
	v_cndmask_b32_e64 v5, 0, 4, vcc
	v_add_lshl_u32 v5, v5, v2, 2
	ds_bpermute_b32 v3, v5, v1
	v_cmp_gt_u32_e32 vcc, 62, v4
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v1, v1, v3
	v_cndmask_b32_e64 v5, 0, 2, vcc
	v_add_lshl_u32 v5, v5, v2, 2
	ds_bpermute_b32 v3, v5, v1
	v_cmp_ne_u32_e32 vcc, 63, v4
	s_nop 1
	v_addc_co_u32_e32 v4, vcc, 0, v2, vcc
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v2, v1, v3
	v_lshlrev_b32_e32 v1, 2, v4
	ds_bpermute_b32 v3, v1, v2
	v_and_b32_e32 v4, 63, v0
	v_lshrrev_b32_e32 v1, 6, v0
	v_cmp_eq_u32_e64 s[6:7], 0, v4
	s_and_saveexec_b64 s[4:5], s[6:7]
	s_cbranch_execz .LBB0_3
; %bb.2:
	v_lshlrev_b32_e32 v4, 2, v1
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v2, v2, v3
	ds_write_b32 v4, v2 offset:512
.LBB0_3:
	s_or_b64 exec, exec, s[4:5]
	v_mov_b32_e32 v2, 0
	v_cmp_eq_u32_e64 s[4:5], 0, v0
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_and_saveexec_b64 s[10:11], s[4:5]
	s_cbranch_execz .LBB0_5
; %bb.4:
	ds_read_b64 v[4:5], v2 offset:512
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v3, v5, v4
	ds_write_b32 v2, v3 offset:520
.LBB0_5:
	s_or_b64 exec, exec, s[10:11]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b32 v2, v2 offset:520
	s_mul_hi_i32 s3, s2, 52
	s_mul_i32 s2, s2, 52
	s_add_u32 s2, s8, s2
	s_mov_b32 s8, 0x26901d7d
	s_waitcnt lgkmcnt(0)
	v_sqrt_f32_e32 v126, v2
	s_addc_u32 s3, s9, s3
	v_cmp_le_f32_e32 vcc, s8, v126
	s_mov_b64 s[8:9], -1
	s_cbranch_vccz .LBB0_15
; %bb.6:
	s_load_dwordx2 s[0:1], s[0:1], 0x8
	v_lshlrev_b32_e32 v6, 9, v0
	v_mov_b32_e32 v127, 0
	s_waitcnt lgkmcnt(0)
	global_load_dwordx4 v[10:13], v6, s[0:1] offset:48
	global_load_dwordx4 v[2:5], v6, s[0:1] offset:32
	s_waitcnt vmcnt(0)
	scratch_store_dwordx4 off, v[2:5], off offset:80 ; 16-byte Folded Spill
	global_load_dwordx4 v[2:5], v6, s[0:1] offset:16
	s_waitcnt vmcnt(0)
	scratch_store_dwordx4 off, v[2:5], off offset:208 ; 16-byte Folded Spill
	global_load_dwordx4 v[2:5], v6, s[0:1]
	s_waitcnt vmcnt(0)
	scratch_store_dwordx4 off, v[2:5], off offset:96 ; 16-byte Folded Spill
	ds_read_b128 v[2:5], v127
	s_waitcnt lgkmcnt(0)
	scratch_store_dwordx4 off, v[2:5], off offset:112 ; 16-byte Folded Spill
	ds_read_b128 v[42:45], v127 offset:16
	ds_read_b128 v[2:5], v127 offset:32
	ds_read_b128 v[122:125], v127 offset:48
	s_waitcnt lgkmcnt(1)
	scratch_store_dwordx4 off, v[2:5], off offset:544 ; 16-byte Folded Spill
	global_load_dwordx4 v[32:35], v6, s[0:1] offset:112
	global_load_dwordx4 v[2:5], v6, s[0:1] offset:96
	s_waitcnt vmcnt(0)
	scratch_store_dwordx4 off, v[2:5], off offset:128 ; 16-byte Folded Spill
	global_load_dwordx4 v[54:57], v6, s[0:1] offset:80
	global_load_dwordx4 v[2:5], v6, s[0:1] offset:64
	s_waitcnt vmcnt(0)
	scratch_store_dwordx4 off, v[2:5], off offset:464 ; 16-byte Folded Spill
	ds_read_b128 v[62:65], v127 offset:80
	ds_read_b128 v[2:5], v127 offset:64
	s_waitcnt lgkmcnt(0)
	scratch_store_dwordx4 off, v[2:5], off offset:512 ; 16-byte Folded Spill
	ds_read_b128 v[58:61], v127 offset:112
	ds_read_b128 v[2:5], v127 offset:96
	s_waitcnt lgkmcnt(0)
	scratch_store_dwordx4 off, v[2:5], off offset:144 ; 16-byte Folded Spill
	global_load_dwordx4 v[2:5], v6, s[0:1] offset:176
	s_waitcnt vmcnt(0)
	scratch_store_dwordx4 off, v[2:5], off offset:592 ; 16-byte Folded Spill
	global_load_dwordx4 v[2:5], v6, s[0:1] offset:160
	s_waitcnt vmcnt(0)
	scratch_store_dwordx4 off, v[2:5], off offset:160 ; 16-byte Folded Spill
	global_load_dwordx4 v[70:73], v6, s[0:1] offset:144
	global_load_dwordx4 v[2:5], v6, s[0:1] offset:128
	s_waitcnt vmcnt(0)
	scratch_store_dwordx4 off, v[2:5], off offset:576 ; 16-byte Folded Spill
	ds_read_b128 v[104:107], v127 offset:144
	ds_read_b128 v[2:5], v127 offset:128
	s_waitcnt lgkmcnt(0)
	scratch_store_dwordx4 off, v[2:5], off offset:608 ; 16-byte Folded Spill
	ds_read_b128 v[2:5], v127 offset:176
	s_waitcnt lgkmcnt(0)
	scratch_store_dwordx4 off, v[2:5], off offset:416 ; 16-byte Folded Spill
	ds_read_b128 v[2:5], v127 offset:160
	s_waitcnt lgkmcnt(0)
	scratch_store_dwordx4 off, v[2:5], off offset:176 ; 16-byte Folded Spill
	global_load_dwordx4 v[86:89], v6, s[0:1] offset:240
	global_load_dwordx4 v[2:5], v6, s[0:1] offset:224
	s_waitcnt vmcnt(0)
	scratch_store_dwordx4 off, v[2:5], off  ; 16-byte Folded Spill
	global_load_dwordx4 v[26:29], v6, s[0:1] offset:208
	global_load_dwordx4 v[2:5], v6, s[0:1] offset:192
	s_waitcnt vmcnt(0)
	scratch_store_dwordx4 off, v[2:5], off offset:624 ; 16-byte Folded Spill
	ds_read_b128 v[96:99], v127 offset:208
	ds_read_b128 v[2:5], v127 offset:192
	s_waitcnt lgkmcnt(0)
	scratch_store_dwordx4 off, v[2:5], off offset:480 ; 16-byte Folded Spill
	ds_read_b128 v[2:5], v127 offset:240
	s_waitcnt lgkmcnt(0)
	scratch_store_dwordx4 off, v[2:5], off offset:432 ; 16-byte Folded Spill
	ds_read_b128 v[2:5], v127 offset:224
	s_waitcnt lgkmcnt(0)
	scratch_store_dwordx4 off, v[2:5], off offset:192 ; 16-byte Folded Spill
	global_load_dwordx4 v[100:103], v6, s[0:1] offset:304
	global_load_dwordx4 v[2:5], v6, s[0:1] offset:288
	s_waitcnt vmcnt(0)
	scratch_store_dwordx4 off, v[2:5], off offset:16 ; 16-byte Folded Spill
	global_load_dwordx4 v[66:69], v6, s[0:1] offset:272
	global_load_dwordx4 v[2:5], v6, s[0:1] offset:256
	s_waitcnt vmcnt(0)
	scratch_store_dwordx4 off, v[2:5], off offset:32 ; 16-byte Folded Spill
	ds_read_b128 v[90:93], v127 offset:272
	ds_read_b128 v[2:5], v127 offset:256
	s_waitcnt lgkmcnt(0)
	scratch_store_dwordx4 off, v[2:5], off offset:48 ; 16-byte Folded Spill
	ds_read_b128 v[82:85], v127 offset:304
	ds_read_b128 v[2:5], v127 offset:288
	s_waitcnt lgkmcnt(0)
	scratch_store_dwordx4 off, v[2:5], off offset:224 ; 16-byte Folded Spill
	global_load_dwordx4 v[50:53], v6, s[0:1] offset:368
	global_load_dwordx4 v[2:5], v6, s[0:1] offset:352
	s_waitcnt vmcnt(0)
	scratch_store_dwordx4 off, v[2:5], off offset:352 ; 16-byte Folded Spill
	global_load_dwordx4 v[78:81], v6, s[0:1] offset:336
	global_load_dwordx4 v[2:5], v6, s[0:1] offset:320
	s_waitcnt vmcnt(0)
	scratch_store_dwordx4 off, v[2:5], off offset:384 ; 16-byte Folded Spill
	ds_read_b128 v[36:39], v127 offset:336
	ds_read_b128 v[2:5], v127 offset:320
	s_waitcnt lgkmcnt(0)
	scratch_store_dwordx4 off, v[2:5], off offset:64 ; 16-byte Folded Spill
	ds_read_b128 v[108:111], v127 offset:368
	ds_read_b128 v[2:5], v127 offset:352
	s_waitcnt lgkmcnt(0)
	scratch_store_dwordx4 off, v[2:5], off offset:400 ; 16-byte Folded Spill
	global_load_dwordx4 v[2:5], v6, s[0:1] offset:432
	s_waitcnt vmcnt(0)
	scratch_store_dwordx4 off, v[2:5], off offset:448 ; 16-byte Folded Spill
	global_load_dwordx4 v[2:5], v6, s[0:1] offset:416
	s_waitcnt vmcnt(0)
	scratch_store_dwordx4 off, v[2:5], off offset:256 ; 16-byte Folded Spill
	global_load_dwordx4 v[14:17], v6, s[0:1] offset:400
	global_load_dwordx4 v[2:5], v6, s[0:1] offset:384
	s_waitcnt vmcnt(0)
	scratch_store_dwordx4 off, v[2:5], off offset:272 ; 16-byte Folded Spill
	ds_read_b128 v[46:49], v127 offset:400
	ds_read_b128 v[2:5], v127 offset:384
	s_waitcnt lgkmcnt(0)
	scratch_store_dwordx4 off, v[2:5], off offset:368 ; 16-byte Folded Spill
	ds_read_b128 v[2:5], v127 offset:432
	s_waitcnt lgkmcnt(0)
	scratch_store_dwordx4 off, v[2:5], off offset:528 ; 16-byte Folded Spill
	ds_read_b128 v[2:5], v127 offset:416
	s_waitcnt lgkmcnt(0)
	scratch_store_dwordx4 off, v[2:5], off offset:336 ; 16-byte Folded Spill
	global_load_dwordx4 v[2:5], v6, s[0:1] offset:496
	s_waitcnt vmcnt(0)
	scratch_store_dwordx4 off, v[2:5], off offset:496 ; 16-byte Folded Spill
	global_load_dwordx4 v[2:5], v6, s[0:1] offset:480
	s_waitcnt vmcnt(0)
	scratch_store_dwordx4 off, v[2:5], off offset:240 ; 16-byte Folded Spill
	global_load_dwordx4 v[20:23], v6, s[0:1] offset:464
	global_load_dwordx4 v[2:5], v6, s[0:1] offset:448
	s_getpc_b64 s[0:1]
	s_add_u32 s0, s0, d_cb3@rel32@lo+4
	s_addc_u32 s1, s1, d_cb3@rel32@hi+12
	s_load_dwordx8 s[8:15], s[0:1], 0x0
	s_waitcnt vmcnt(0)
	scratch_store_dwordx4 off, v[2:5], off offset:288 ; 16-byte Folded Spill
	ds_read_b128 v[74:77], v127 offset:464
	ds_read_b128 v[2:5], v127 offset:448
	s_waitcnt lgkmcnt(0)
	scratch_store_dwordx4 off, v[2:5], off offset:320 ; 16-byte Folded Spill
	ds_read_b128 v[2:5], v127 offset:496
	s_waitcnt lgkmcnt(0)
	scratch_store_dwordx4 off, v[2:5], off offset:560 ; 16-byte Folded Spill
	ds_read_b128 v[2:5], v127 offset:480
	s_waitcnt lgkmcnt(0)
	scratch_store_dwordx4 off, v[2:5], off offset:304 ; 16-byte Folded Spill
	s_and_saveexec_b64 s[0:1], s[4:5]
	s_cbranch_execz .LBB0_8
; %bb.7:
	global_store_dword v127, v126, s[2:3]
.LBB0_8:
	s_or_b64 exec, exec, s[0:1]
	scratch_load_dwordx4 v[114:117], off, off offset:208 ; 16-byte Folded Reload
	v_pk_mul_f32 v[18:19], v[122:123], v[10:11]
	v_mov_b64_e32 v[6:7], v[104:105]
	v_mov_b64_e32 v[8:9], v[106:107]
	v_mov_b64_e32 v[122:123], v[12:13]
	s_waitcnt vmcnt(0)
	v_pk_fma_f32 v[10:11], v[42:43], v[114:115], v[18:19]
	s_nop 0
	v_pk_fma_f32 v[10:11], v[62:63], v[54:55], v[10:11]
	v_mov_b64_e32 v[114:115], v[98:99]
	v_pk_fma_f32 v[10:11], v[58:59], v[32:33], v[10:11]
	scratch_load_dwordx4 v[104:107], off, off offset:416 ; 16-byte Folded Reload
	scratch_load_dwordx4 v[30:33], off, off offset:592 ; 16-byte Folded Reload
	v_mov_b64_e32 v[112:113], v[96:97]
	scratch_load_dwordx4 v[94:97], off, off offset:432 ; 16-byte Folded Reload
	v_pk_fma_f32 v[10:11], v[6:7], v[70:71], v[10:11]
	v_mov_b64_e32 v[70:71], v[68:69]
	v_mov_b64_e32 v[68:69], v[66:67]
	v_mov_b64_e32 v[62:63], v[38:39]
	s_waitcnt vmcnt(1)
	v_pk_fma_f32 v[10:11], v[104:105], v[30:31], v[10:11]
	v_mov_b64_e32 v[30:31], v[28:29]
	v_mov_b64_e32 v[28:29], v[26:27]
	v_pk_fma_f32 v[10:11], v[112:113], v[28:29], v[10:11]
	scratch_load_dwordx4 v[26:29], off, off offset:528 ; 16-byte Folded Reload
	s_waitcnt vmcnt(1)
	v_pk_fma_f32 v[10:11], v[94:95], v[86:87], v[10:11]
	v_mov_b64_e32 v[94:95], v[92:93]
	v_mov_b64_e32 v[92:93], v[90:91]
	v_mov_b64_e32 v[86:87], v[84:85]
	v_pk_fma_f32 v[10:11], v[92:93], v[68:69], v[10:11]
	v_mov_b64_e32 v[84:85], v[82:83]
	v_mov_b64_e32 v[82:83], v[80:81]
	v_pk_fma_f32 v[10:11], v[84:85], v[100:101], v[10:11]
	v_mov_b64_e32 v[80:81], v[78:79]
	v_pk_fma_f32 v[10:11], v[36:37], v[80:81], v[10:11]
	v_mov_b64_e32 v[68:69], v[22:23]
	v_pk_fma_f32 v[10:11], v[108:109], v[50:51], v[10:11]
	v_mov_b64_e32 v[108:109], v[48:49]
	v_pk_fma_f32 v[10:11], v[46:47], v[14:15], v[10:11]
	scratch_load_dwordx4 v[46:49], off, off offset:448 ; 16-byte Folded Reload
	v_mov_b64_e32 v[66:67], v[20:21]
	scratch_load_dwordx4 v[78:81], off, off offset:496 ; 16-byte Folded Reload
	scratch_load_dwordx4 v[22:25], off, off offset:560 ; 16-byte Folded Reload
	scratch_load_dwordx4 v[90:93], off, off offset:80 ; 16-byte Folded Reload
	scratch_load_dwordx4 v[4:7], off, off offset:544 ; 16-byte Folded Reload
	scratch_load_dwordx4 v[40:43], off, off offset:96 ; 16-byte Folded Reload
	scratch_load_dwordx4 v[18:21], off, off offset:112 ; 16-byte Folded Reload
	scratch_load_dwordx4 v[98:101], off, off offset:128 ; 16-byte Folded Reload
	v_mov_b64_e32 v[104:105], v[16:17]
	v_mov_b64_e32 v[58:59], v[82:83]
	scratch_load_dwordx4 v[36:39], off, off offset:512 ; 16-byte Folded Reload
	s_waitcnt vmcnt(8)
	v_pk_fma_f32 v[10:11], v[26:27], v[46:47], v[10:11]
	s_nop 0
	v_pk_fma_f32 v[2:3], v[74:75], v[66:67], v[10:11]
	v_mov_b64_e32 v[66:67], v[32:33]
	s_waitcnt vmcnt(6)
	v_pk_fma_f32 v[2:3], v[22:23], v[78:79], v[2:3]
	s_waitcnt vmcnt(4)
	v_mov_b64_e32 v[50:51], v[6:7]
	v_pk_fma_f32 v[2:3], v[4:5], v[90:91], v[2:3]
	scratch_load_dwordx4 v[4:7], off, off offset:464 ; 16-byte Folded Reload
	s_waitcnt vmcnt(3)
	v_pk_fma_f32 v[2:3], v[18:19], v[40:41], v[2:3]
	v_mov_b64_e32 v[74:75], v[30:31]
	s_waitcnt vmcnt(1)
	v_mov_b64_e32 v[40:41], v[38:39]
	s_waitcnt vmcnt(0)
	v_pk_fma_f32 v[2:3], v[36:37], v[4:5], v[2:3]
	scratch_load_dwordx4 v[36:39], off, off offset:144 ; 16-byte Folded Reload
	v_mov_b64_e32 v[54:55], v[6:7]
	scratch_load_dwordx4 v[4:7], off, off offset:576 ; 16-byte Folded Reload
	scratch_load_dwordx4 v[16:19], off, off offset:608 ; 16-byte Folded Reload
	scratch_load_dwordx4 v[118:121], off, off offset:160 ; 16-byte Folded Reload
	s_waitcnt vmcnt(3)
	v_pk_fma_f32 v[2:3], v[36:37], v[98:99], v[2:3]
	s_waitcnt vmcnt(2)
	v_mov_b64_e32 v[46:47], v[6:7]
	s_waitcnt vmcnt(1)
	v_pk_fma_f32 v[2:3], v[16:17], v[4:5], v[2:3]
	v_mov_b64_e32 v[22:23], v[18:19]
	scratch_load_dwordx4 v[16:19], off, off offset:176 ; 16-byte Folded Reload
	scratch_load_dwordx4 v[4:7], off, off offset:480 ; 16-byte Folded Reload
	scratch_load_dwordx4 v[82:85], off, off offset:624 ; 16-byte Folded Reload
	s_waitcnt vmcnt(1)
	v_mov_b64_e32 v[36:37], v[6:7]
	v_pk_fma_f32 v[2:3], v[16:17], v[118:119], v[2:3]
	s_waitcnt vmcnt(0)
	v_mov_b64_e32 v[26:27], v[84:85]
	v_pk_fma_f32 v[2:3], v[4:5], v[82:83], v[2:3]
	scratch_load_dwordx4 v[82:85], off, off offset:192 ; 16-byte Folded Reload
	scratch_load_dwordx4 v[4:7], off, off   ; 16-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_pk_fma_f32 v[2:3], v[82:83], v[4:5], v[2:3]
	scratch_load_dwordx4 v[4:7], off, off offset:32 ; 16-byte Folded Reload
	scratch_load_dwordx4 v[10:13], off, off offset:48 ; 16-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_pk_fma_f32 v[2:3], v[10:11], v[4:5], v[2:3]
	scratch_load_dwordx4 v[30:33], off, off offset:224 ; 16-byte Folded Reload
	scratch_load_dwordx4 v[4:7], off, off offset:16 ; 16-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_pk_fma_f32 v[2:3], v[30:31], v[4:5], v[2:3]
	scratch_load_dwordx4 v[4:7], off, off offset:384 ; 16-byte Folded Reload
	scratch_load_dwordx4 v[10:13], off, off offset:64 ; 16-byte Folded Reload
	s_waitcnt vmcnt(1)
	v_mov_b64_e32 v[118:119], v[6:7]
	s_waitcnt vmcnt(0)
	v_pk_fma_f32 v[2:3], v[10:11], v[4:5], v[2:3]
	scratch_load_dwordx4 v[4:7], off, off offset:352 ; 16-byte Folded Reload
	scratch_load_dwordx4 v[10:13], off, off offset:400 ; 16-byte Folded Reload
	s_waitcnt vmcnt(1)
	v_mov_b64_e32 v[98:99], v[6:7]
	s_waitcnt vmcnt(0)
	v_pk_fma_f32 v[2:3], v[10:11], v[4:5], v[2:3]
	v_mov_b64_e32 v[112:113], v[12:13]
	scratch_load_dwordx4 v[4:7], off, off offset:272 ; 16-byte Folded Reload
	scratch_load_dwordx4 v[10:13], off, off offset:368 ; 16-byte Folded Reload
	s_waitcnt vmcnt(1)
	v_mov_b64_e32 v[82:83], v[6:7]
	s_waitcnt vmcnt(0)
	v_pk_fma_f32 v[2:3], v[10:11], v[4:5], v[2:3]
	v_mov_b64_e32 v[90:91], v[12:13]
	scratch_load_dwordx4 v[4:7], off, off offset:256 ; 16-byte Folded Reload
	scratch_load_dwordx4 v[10:13], off, off offset:336 ; 16-byte Folded Reload
	s_waitcnt vmcnt(1)
	v_mov_b64_e32 v[30:31], v[6:7]
	s_waitcnt vmcnt(0)
	v_pk_fma_f32 v[2:3], v[10:11], v[4:5], v[2:3]
	v_mov_b64_e32 v[78:79], v[12:13]
	scratch_load_dwordx4 v[4:7], off, off offset:288 ; 16-byte Folded Reload
	scratch_load_dwordx4 v[12:15], off, off offset:320 ; 16-byte Folded Reload
	s_waitcnt vmcnt(1)
	v_mov_b64_e32 v[16:17], v[6:7]
	s_waitcnt vmcnt(0)
	v_pk_fma_f32 v[2:3], v[12:13], v[4:5], v[2:3]
	scratch_load_dwordx4 v[4:7], off, off offset:240 ; 16-byte Folded Reload
	scratch_load_dwordx4 v[10:13], off, off offset:304 ; 16-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_pk_fma_f32 v[2:3], v[10:11], v[4:5], v[2:3]
	s_nop 0
	v_pk_fma_f32 v[2:3], v[124:125], v[122:123], v[2:3]
	v_rcp_f32_e32 v4, v126
	v_pk_fma_f32 v[2:3], v[44:45], v[116:117], v[2:3]
	v_lshlrev_b32_e32 v126, 3, v1
	v_pk_fma_f32 v[2:3], v[64:65], v[56:57], v[2:3]
	s_nop 0
	v_pk_fma_f32 v[2:3], v[60:61], v[34:35], v[2:3]
	s_nop 0
	v_pk_fma_f32 v[2:3], v[8:9], v[72:73], v[2:3]
	scratch_load_dwordx4 v[8:11], off, off  ; 16-byte Folded Reload
	v_pk_fma_f32 v[2:3], v[106:107], v[66:67], v[2:3]
	s_nop 0
	v_pk_fma_f32 v[2:3], v[114:115], v[74:75], v[2:3]
	s_nop 0
	v_pk_fma_f32 v[2:3], v[96:97], v[88:89], v[2:3]
	s_nop 0
	v_pk_fma_f32 v[2:3], v[94:95], v[70:71], v[2:3]
	s_nop 0
	v_pk_fma_f32 v[2:3], v[86:87], v[102:103], v[2:3]
	s_nop 0
	v_pk_fma_f32 v[2:3], v[62:63], v[58:59], v[2:3]
	s_nop 0
	v_pk_fma_f32 v[2:3], v[110:111], v[52:53], v[2:3]
	s_nop 0
	v_pk_fma_f32 v[2:3], v[108:109], v[104:105], v[2:3]
	s_nop 0
	v_pk_fma_f32 v[2:3], v[28:29], v[48:49], v[2:3]
	s_nop 0
	v_pk_fma_f32 v[2:3], v[76:77], v[68:69], v[2:3]
	s_nop 0
	v_pk_fma_f32 v[2:3], v[24:25], v[80:81], v[2:3]
	s_nop 0
	v_pk_fma_f32 v[2:3], v[50:51], v[92:93], v[2:3]
	s_nop 0
	v_pk_fma_f32 v[2:3], v[20:21], v[42:43], v[2:3]
	s_nop 0
	v_pk_fma_f32 v[2:3], v[40:41], v[54:55], v[2:3]
	s_nop 0
	v_pk_fma_f32 v[2:3], v[38:39], v[100:101], v[2:3]
	s_nop 0
	v_pk_fma_f32 v[2:3], v[22:23], v[46:47], v[2:3]
	s_nop 0
	v_pk_fma_f32 v[2:3], v[18:19], v[120:121], v[2:3]
	s_nop 0
	v_pk_fma_f32 v[2:3], v[36:37], v[26:27], v[2:3]
	s_waitcnt vmcnt(0)
	v_pk_fma_f32 v[2:3], v[84:85], v[10:11], v[2:3]
	scratch_load_dwordx4 v[8:11], off, off offset:32 ; 16-byte Folded Reload
	scratch_load_dwordx4 v[18:21], off, off offset:48 ; 16-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_pk_fma_f32 v[2:3], v[20:21], v[10:11], v[2:3]
	scratch_load_dwordx4 v[8:11], off, off offset:16 ; 16-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_pk_fma_f32 v[2:3], v[32:33], v[10:11], v[2:3]
	scratch_load_dwordx4 v[8:11], off, off offset:64 ; 16-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_pk_fma_f32 v[2:3], v[10:11], v[118:119], v[2:3]
	s_nop 0
	v_pk_fma_f32 v[2:3], v[112:113], v[98:99], v[2:3]
	s_nop 0
	v_pk_fma_f32 v[2:3], v[90:91], v[82:83], v[2:3]
	s_nop 0
	v_pk_fma_f32 v[2:3], v[78:79], v[30:31], v[2:3]
	s_nop 0
	v_pk_fma_f32 v[2:3], v[14:15], v[16:17], v[2:3]
	s_nop 0
	v_pk_fma_f32 v[2:3], v[12:13], v[6:7], v[2:3]
	s_nop 0
	v_add_f32_e32 v2, v2, v3
	v_fma_f32 v3, v4, v2, -s8
	v_mul_f32_e32 v3, v3, v3
	v_fma_f32 v5, v4, v2, -s9
	v_min_f32_e32 v3, 0x7149f2ca, v3
	v_mul_f32_e32 v5, v5, v5
	v_cmp_lt_f32_e32 vcc, v5, v3
	v_fma_f32 v7, v4, v2, -s11
	v_mul_f32_e32 v7, v7, v7
	v_cndmask_b32_e32 v3, v3, v5, vcc
	v_fma_f32 v5, v4, v2, -s10
	v_mul_f32_e32 v5, v5, v5
	v_min_f32_e32 v6, v5, v3
	v_fma_f32 v9, v4, v2, -s12
	v_fma_f32 v11, v4, v2, -s13
	v_fma_f32 v13, v4, v2, -s14
	v_fma_f32 v2, v4, v2, -s15
	v_cndmask_b32_e64 v4, 0, 1, vcc
	v_cmp_ge_f32_e32 vcc, v5, v3
	v_min_f32_e32 v8, v7, v6
	v_mul_f32_e32 v9, v9, v9
	v_cndmask_b32_e32 v3, 2, v4, vcc
	v_cmp_ge_f32_e32 vcc, v7, v6
	v_min_f32_e32 v10, v9, v8
	v_mul_f32_e32 v11, v11, v11
	v_cndmask_b32_e32 v3, 3, v3, vcc
	v_cmp_ge_f32_e32 vcc, v9, v8
	v_min_f32_e32 v12, v11, v10
	v_mul_f32_e32 v13, v13, v13
	v_cndmask_b32_e32 v3, 4, v3, vcc
	v_cmp_ge_f32_e32 vcc, v11, v10
	v_min_f32_e32 v14, v13, v12
	v_mul_f32_e32 v2, v2, v2
	v_cndmask_b32_e32 v3, 5, v3, vcc
	v_cmp_ge_f32_e32 vcc, v13, v12
	s_nop 1
	v_cndmask_b32_e32 v3, 6, v3, vcc
	v_cmp_ge_f32_e32 vcc, v2, v14
	s_nop 1
	v_cndmask_b32_e32 v4, 7, v3, vcc
	v_and_b32_e32 v1, 1, v4
	v_lshl_add_u64 v[2:3], s[2:3], 0, v[126:127]
	v_cmp_ne_u32_e64 s[8:9], 0, v1
	s_and_saveexec_b64 s[0:1], s[6:7]
	s_cbranch_execz .LBB0_10
; %bb.9:
	v_mov_b64_e32 v[6:7], s[8:9]
	global_store_dwordx2 v[2:3], v[6:7], off offset:4
.LBB0_10:
	s_or_b64 exec, exec, s[0:1]
	v_bfe_u32 v1, v4, 1, 1
	v_cmp_ne_u32_e64 s[8:9], 0, v1
	s_and_saveexec_b64 s[0:1], s[6:7]
	s_cbranch_execz .LBB0_12
; %bb.11:
	v_mov_b64_e32 v[6:7], s[8:9]
	global_store_dwordx2 v[2:3], v[6:7], off offset:20
.LBB0_12:
	s_or_b64 exec, exec, s[0:1]
	v_bfe_u32 v1, v4, 2, 1
	v_cmp_ne_u32_e64 s[8:9], 0, v1
	s_and_saveexec_b64 s[0:1], s[6:7]
	s_cbranch_execz .LBB0_14
; %bb.13:
	v_mov_b64_e32 v[4:5], s[8:9]
	global_store_dwordx2 v[2:3], v[4:5], off offset:36
.LBB0_14:                               ; %Flow
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[8:9], 0
.LBB0_15:                               ; %Flow82
	s_and_b64 vcc, exec, s[8:9]
	s_cbranch_vccz .LBB0_20
; %bb.16:
	s_and_saveexec_b64 s[0:1], s[4:5]
	s_cbranch_execz .LBB0_18
; %bb.17:
	v_mov_b32_e32 v1, 0
	global_store_dword v1, v1, s[2:3]
.LBB0_18:
	s_or_b64 exec, exec, s[0:1]
	v_cmp_gt_u32_e32 vcc, 6, v0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_20
; %bb.19:
	v_lshlrev_b32_e32 v2, 3, v0
	v_mov_b32_e32 v0, 0
	v_mov_b32_e32 v1, v0
	global_store_dwordx2 v2, v[0:1], s[2:3] offset:4
.LBB0_20:                               ; %.loopexit
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z23tqm_quantize_kernel_tq3PKfS0_P16block_tq3_mi300xi
		.amdhsa_group_segment_fixed_size 524
		.amdhsa_private_segment_fixed_size 644
		.amdhsa_kernarg_size 28
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 0
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 128
		.amdhsa_next_free_sgpr 16
		.amdhsa_accum_offset 128
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_tg_split 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end0:
	.size	_Z23tqm_quantize_kernel_tq3PKfS0_P16block_tq3_mi300xi, .Lfunc_end0-_Z23tqm_quantize_kernel_tq3PKfS0_P16block_tq3_mi300xi
                                        ; -- End function
	.set _Z23tqm_quantize_kernel_tq3PKfS0_P16block_tq3_mi300xi.num_vgpr, 128
	.set _Z23tqm_quantize_kernel_tq3PKfS0_P16block_tq3_mi300xi.num_agpr, 0
	.set _Z23tqm_quantize_kernel_tq3PKfS0_P16block_tq3_mi300xi.numbered_sgpr, 16
	.set _Z23tqm_quantize_kernel_tq3PKfS0_P16block_tq3_mi300xi.num_named_barrier, 0
	.set _Z23tqm_quantize_kernel_tq3PKfS0_P16block_tq3_mi300xi.private_seg_size, 644
	.set _Z23tqm_quantize_kernel_tq3PKfS0_P16block_tq3_mi300xi.uses_vcc, 1
	.set _Z23tqm_quantize_kernel_tq3PKfS0_P16block_tq3_mi300xi.uses_flat_scratch, 0
	.set _Z23tqm_quantize_kernel_tq3PKfS0_P16block_tq3_mi300xi.has_dyn_sized_stack, 0
	.set _Z23tqm_quantize_kernel_tq3PKfS0_P16block_tq3_mi300xi.has_recursion, 0
	.set _Z23tqm_quantize_kernel_tq3PKfS0_P16block_tq3_mi300xi.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 3220
; TotalNumSgprs: 22
; NumVgprs: 128
; NumAgprs: 0
; TotalNumVgprs: 128
; ScratchSize: 644
; MemoryBound: 1
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 524 bytes/workgroup (compile time only)
; SGPRBlocks: 2
; VGPRBlocks: 15
; NumSGPRsForWavesPerEU: 22
; NumVGPRsForWavesPerEU: 128
; AccumOffset: 128
; Occupancy: 4
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 1
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 31
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.protected	_Z25tqm_dequantize_kernel_tq3PK16block_tq3_mi300xPKfPfi ; -- Begin function _Z25tqm_dequantize_kernel_tq3PK16block_tq3_mi300xPKfPfi
	.globl	_Z25tqm_dequantize_kernel_tq3PK16block_tq3_mi300xPKfPfi
	.p2align	8
	.type	_Z25tqm_dequantize_kernel_tq3PK16block_tq3_mi300xPKfPfi,@function
_Z25tqm_dequantize_kernel_tq3PK16block_tq3_mi300xPKfPfi: ; @_Z25tqm_dequantize_kernel_tq3PK16block_tq3_mi300xPKfPfi
; %bb.0:
	s_load_dword s3, s[0:1], 0x18
	s_waitcnt lgkmcnt(0)
	s_cmp_ge_i32 s2, s3
	s_cbranch_scc1 .LBB1_6
; %bb.1:
	s_load_dwordx2 s[6:7], s[0:1], 0x0
	s_load_dwordx2 s[4:5], s[0:1], 0x10
	s_mul_i32 s8, s2, 52
	s_mul_hi_i32 s3, s2, 52
	v_cmp_eq_u32_e32 vcc, 0, v0
	s_waitcnt lgkmcnt(0)
	s_add_u32 s6, s6, s8
	s_addc_u32 s7, s7, s3
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB1_3
; %bb.2:
	v_mov_b32_e32 v1, 0
	global_load_dword v2, v1, s[6:7]
	s_waitcnt vmcnt(0)
	ds_write_b32 v1, v2 offset:512
.LBB1_3:
	s_or_b64 exec, exec, s[8:9]
	v_mov_b32_e32 v1, 0
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b32 v2, v1 offset:512
	s_mov_b32 s3, 0x26901d7d
	s_waitcnt lgkmcnt(0)
	v_cmp_lt_f32_e64 s[8:9], |v2|, s3
	s_and_b64 vcc, exec, s[8:9]
	s_cbranch_vccnz .LBB1_5
; %bb.4:                                ; %.preheader
	v_lshrrev_b32_e32 v1, 3, v0
	v_and_b32_e32 v1, 0x78, v1
	global_load_dwordx2 v[2:3], v1, s[6:7] offset:4
	global_load_dwordx2 v[4:5], v1, s[6:7] offset:20
	global_load_dwordx2 v[6:7], v1, s[6:7] offset:36
	v_mov_b32_e32 v11, 0
	s_getpc_b64 s[6:7]
	s_add_u32 s6, s6, d_cb3@rel32@lo+4
	s_addc_u32 s7, s7, d_cb3@rel32@hi+12
	s_load_dwordx2 s[0:1], s[0:1], 0x8
	s_waitcnt vmcnt(2)
	v_lshrrev_b64 v[2:3], v0, v[2:3]
	s_waitcnt vmcnt(1)
	v_lshrrev_b64 v[4:5], v0, v[4:5]
	v_and_b32_e32 v1, 1, v2
	s_waitcnt vmcnt(0)
	v_lshrrev_b64 v[6:7], v0, v[6:7]
	v_lshlrev_b32_e32 v4, 3, v4
	v_lshlrev_b32_e32 v10, 2, v1
	v_lshlrev_b32_e32 v5, 4, v6
	v_lshl_add_u64 v[2:3], s[6:7], 0, v[10:11]
	v_and_b32_e32 v10, 8, v4
	v_lshl_add_u64 v[2:3], v[2:3], 0, v[10:11]
	v_and_b32_e32 v10, 16, v5
	v_lshl_add_u64 v[2:3], v[2:3], 0, v[10:11]
	global_load_dword v2, v[2:3], off
	v_lshlrev_b32_e32 v1, 2, v0
	v_add_u32_e32 v3, 0x200, v1
	v_add_u32_e32 v4, 0x400, v1
	v_add_u32_e32 v5, 0x600, v1
	v_add_u32_e32 v6, 0x800, v1
	v_add_u32_e32 v7, 0xa00, v1
	v_add_u32_e32 v8, 0xc00, v1
	v_add_u32_e32 v9, 0xe00, v1
	v_add_u32_e32 v10, 0x3400, v1
	v_add_u32_e32 v31, 0x3600, v1
	v_add_u32_e32 v32, 0x3800, v1
	v_add_u32_e32 v33, 0x3a00, v1
	v_add_u32_e32 v34, 0x3c00, v1
	v_add_u32_e32 v35, 0x3e00, v1
	v_or_b32_e32 v44, 0x4000, v1
	v_add_u32_e32 v45, 0x4200, v1
	v_add_u32_e32 v46, 0x4400, v1
	v_add_u32_e32 v47, 0x4600, v1
	v_add_u32_e32 v49, 0x4a00, v1
	v_add_u32_e32 v50, 0x4c00, v1
	v_add_u32_e32 v51, 0x4e00, v1
	v_or_b32_e32 v54, 0x5000, v1
	v_add_u32_e32 v55, 0x5200, v1
	v_add_u32_e32 v56, 0x5400, v1
	v_add_u32_e32 v57, 0x5600, v1
	v_add_u32_e32 v58, 0x5800, v1
	v_add_u32_e32 v59, 0x5a00, v1
	v_add_u32_e32 v60, 0x5c00, v1
	v_add_u32_e32 v61, 0x5e00, v1
	v_or_b32_e32 v62, 0x6000, v1
	v_add_u32_e32 v63, 0x6200, v1
	v_add_u32_e32 v64, 0x6400, v1
	v_add_u32_e32 v65, 0x6600, v1
	v_or_b32_e32 v66, 0x7000, v1
	v_add_u32_e32 v67, 0x7200, v1
	v_add_u32_e32 v68, 0x7400, v1
	v_add_u32_e32 v69, 0x7600, v1
	v_or_b32_e32 v70, 0x8000, v1
	v_add_u32_e32 v71, 0x8200, v1
	v_add_u32_e32 v72, 0x8400, v1
	v_add_u32_e32 v73, 0x8600, v1
	v_add_u32_e32 v74, 0x8800, v1
	s_waitcnt vmcnt(0)
	ds_write_b32 v1, v2
	s_waitcnt lgkmcnt(0)
	s_barrier
	global_load_dword v26, v1, s[0:1]
	global_load_dword v27, v3, s[0:1]
	global_load_dword v28, v4, s[0:1]
	global_load_dword v29, v5, s[0:1]
	global_load_dword v24, v6, s[0:1]
	global_load_dword v25, v7, s[0:1]
	global_load_dword v22, v8, s[0:1]
	global_load_dword v23, v9, s[0:1]
	v_or_b32_e32 v2, 0x1000, v1
	v_add_u32_e32 v3, 0x1200, v1
	v_add_u32_e32 v4, 0x1400, v1
	global_load_dword v20, v2, s[0:1]
	global_load_dword v21, v3, s[0:1]
	v_add_u32_e32 v5, 0x1600, v1
	global_load_dword v16, v4, s[0:1]
	global_load_dword v17, v5, s[0:1]
	v_add_u32_e32 v6, 0x1800, v1
	v_add_u32_e32 v2, 0x1a00, v1
	v_add_u32_e32 v3, 0x1c00, v1
	v_add_u32_e32 v4, 0x1e00, v1
	global_load_dword v14, v6, s[0:1]
	global_load_dword v15, v2, s[0:1]
	global_load_dword v12, v3, s[0:1]
	global_load_dword v13, v4, s[0:1]
	v_or_b32_e32 v5, 0x2000, v1
	v_add_u32_e32 v2, 0x2200, v1
	v_add_u32_e32 v3, 0x2400, v1
	global_load_dword v18, v5, s[0:1]
	global_load_dword v19, v2, s[0:1]
	v_add_u32_e32 v2, 0x2600, v1
	global_load_dword v6, v3, s[0:1]
	global_load_dword v7, v2, s[0:1]
	v_add_u32_e32 v2, 0x2800, v1
	v_add_u32_e32 v3, 0x2a00, v1
	v_add_u32_e32 v4, 0x2c00, v1
	v_add_u32_e32 v5, 0x2e00, v1
	global_load_dword v2, v2, s[0:1]
	global_load_dword v3, v3, s[0:1]
	global_load_dword v4, v4, s[0:1]
	global_load_dword v5, v5, s[0:1]
	v_or_b32_e32 v8, 0x3000, v1
	v_add_u32_e32 v9, 0x3200, v1
	global_load_dword v8, v8, s[0:1]
	global_load_dword v9, v9, s[0:1]
	global_load_dword v30, v10, s[0:1]
	global_load_dword v31, v31, s[0:1]
	ds_read_b128 v[40:43], v11
	global_load_dword v38, v32, s[0:1]
	global_load_dword v39, v33, s[0:1]
	global_load_dword v36, v34, s[0:1]
	global_load_dword v37, v35, s[0:1]
	global_load_dword v34, v44, s[0:1]
	global_load_dword v35, v45, s[0:1]
	global_load_dword v32, v46, s[0:1]
	global_load_dword v33, v47, s[0:1]
	ds_read_b128 v[44:47], v11 offset:16
	v_add_u32_e32 v10, 0x4800, v1
	s_waitcnt vmcnt(34) lgkmcnt(1)
	v_pk_mul_f32 v[26:27], v[40:41], v[26:27]
	s_waitcnt vmcnt(32)
	v_pk_fma_f32 v[26:27], v[42:43], v[28:29], v[26:27]
	ds_read_b128 v[40:43], v11 offset:48
	s_waitcnt vmcnt(30) lgkmcnt(1)
	v_pk_fma_f32 v[28:29], v[44:45], v[24:25], v[26:27]
	ds_read_b128 v[24:27], v11 offset:32
	s_waitcnt vmcnt(28)
	v_pk_fma_f32 v[22:23], v[46:47], v[22:23], v[28:29]
	v_add_u32_e32 v44, 0x6800, v1
	v_add_u32_e32 v45, 0x6a00, v1
	v_add_u32_e32 v46, 0x6c00, v1
	s_waitcnt vmcnt(26) lgkmcnt(0)
	v_pk_fma_f32 v[20:21], v[24:25], v[20:21], v[22:23]
	v_add_u32_e32 v47, 0x6e00, v1
	s_waitcnt vmcnt(24)
	v_pk_fma_f32 v[16:17], v[26:27], v[16:17], v[20:21]
	s_waitcnt vmcnt(22)
	v_pk_fma_f32 v[20:21], v[40:41], v[14:15], v[16:17]
	ds_read_b128 v[14:17], v11 offset:64
	s_waitcnt vmcnt(20)
	v_pk_fma_f32 v[12:13], v[42:43], v[12:13], v[20:21]
	ds_read_b128 v[20:23], v11 offset:80
	global_load_dword v48, v10, s[0:1]
	global_load_dword v49, v49, s[0:1]
	global_load_dword v52, v50, s[0:1]
	global_load_dword v53, v51, s[0:1]
	v_add_u32_e32 v40, 0x7800, v1
	s_waitcnt vmcnt(22) lgkmcnt(1)
	v_pk_fma_f32 v[12:13], v[14:15], v[18:19], v[12:13]
	v_add_u32_e32 v41, 0x7a00, v1
	s_waitcnt vmcnt(20)
	v_pk_fma_f32 v[6:7], v[16:17], v[6:7], v[12:13]
	v_add_u32_e32 v42, 0x7c00, v1
	s_waitcnt vmcnt(18) lgkmcnt(0)
	v_pk_fma_f32 v[2:3], v[20:21], v[2:3], v[6:7]
	v_add_u32_e32 v43, 0x7e00, v1
	s_waitcnt vmcnt(16)
	v_pk_fma_f32 v[6:7], v[22:23], v[4:5], v[2:3]
	ds_read_b128 v[2:5], v11 offset:96
	global_load_dword v54, v54, s[0:1]
	global_load_dword v55, v55, s[0:1]
	global_load_dword v24, v56, s[0:1]
	global_load_dword v25, v57, s[0:1]
	ds_read_b128 v[26:29], v11 offset:112
	global_load_dword v22, v58, s[0:1]
	global_load_dword v23, v59, s[0:1]
	global_load_dword v20, v60, s[0:1]
	global_load_dword v21, v61, s[0:1]
	global_load_dword v18, v62, s[0:1]
	global_load_dword v19, v63, s[0:1]
	global_load_dword v16, v64, s[0:1]
	global_load_dword v17, v65, s[0:1]
	s_waitcnt vmcnt(26) lgkmcnt(1)
	v_pk_fma_f32 v[2:3], v[2:3], v[8:9], v[6:7]
	v_add_u32_e32 v10, 0x8a00, v1
	s_waitcnt vmcnt(24)
	v_pk_fma_f32 v[6:7], v[4:5], v[30:31], v[2:3]
	global_load_dword v14, v44, s[0:1]
	global_load_dword v15, v45, s[0:1]
	global_load_dword v12, v46, s[0:1]
	global_load_dword v13, v47, s[0:1]
	global_load_dword v4, v66, s[0:1]
	global_load_dword v5, v67, s[0:1]
	global_load_dword v2, v68, s[0:1]
	global_load_dword v3, v69, s[0:1]
	s_waitcnt vmcnt(30) lgkmcnt(0)
	v_pk_fma_f32 v[26:27], v[26:27], v[38:39], v[6:7]
	ds_read_b128 v[6:9], v11 offset:128
	ds_read_b128 v[44:47], v11 offset:144
	s_waitcnt vmcnt(28)
	v_pk_fma_f32 v[26:27], v[28:29], v[36:37], v[26:27]
	v_or_b32_e32 v30, 0x9000, v1
	v_add_u32_e32 v31, 0x9200, v1
	v_add_u32_e32 v38, 0x9800, v1
	v_add_u32_e32 v39, 0x9a00, v1
	s_waitcnt vmcnt(26) lgkmcnt(1)
	v_pk_fma_f32 v[6:7], v[6:7], v[34:35], v[26:27]
	v_add_u32_e32 v56, 0x8c00, v1
	v_add_u32_e32 v57, 0x8e00, v1
	v_add_u32_e32 v58, 0x9400, v1
	v_add_u32_e32 v59, 0x9600, v1
	v_add_u32_e32 v60, 0x9c00, v1
	v_add_u32_e32 v61, 0x9e00, v1
	s_waitcnt vmcnt(24)
	v_pk_fma_f32 v[50:51], v[8:9], v[32:33], v[6:7]
	v_or_b32_e32 v62, 0xa000, v1
	v_add_u32_e32 v63, 0xa200, v1
	global_load_dword v28, v40, s[0:1]
	global_load_dword v29, v41, s[0:1]
	global_load_dword v8, v42, s[0:1]
	global_load_dword v9, v43, s[0:1]
	global_load_dword v34, v70, s[0:1]
	global_load_dword v35, v71, s[0:1]
	global_load_dword v6, v72, s[0:1]
	global_load_dword v7, v73, s[0:1]
	global_load_dword v32, v74, s[0:1]
	global_load_dword v33, v10, s[0:1]
	global_load_dword v26, v56, s[0:1]
	global_load_dword v27, v57, s[0:1]
	global_load_dword v30, v30, s[0:1]
	global_load_dword v31, v31, s[0:1]
	global_load_dword v36, v58, s[0:1]
	global_load_dword v37, v59, s[0:1]
	global_load_dword v38, v38, s[0:1]
	global_load_dword v39, v39, s[0:1]
	global_load_dword v40, v60, s[0:1]
	global_load_dword v41, v61, s[0:1]
	global_load_dword v42, v62, s[0:1]
	global_load_dword v43, v63, s[0:1]
	v_add_u32_e32 v10, 0xa400, v1
	v_add_u32_e32 v56, 0xa600, v1
	v_add_u32_e32 v57, 0xa800, v1
	v_add_u32_e32 v58, 0xaa00, v1
	v_add_u32_e32 v59, 0xb800, v1
	v_add_u32_e32 v60, 0xba00, v1
	v_or_b32_e32 v61, 0xc000, v1
	v_add_u32_e32 v62, 0xc200, v1
	v_add_u32_e32 v63, 0xc400, v1
	v_add_u32_e32 v64, 0xc600, v1
	v_add_u32_e32 v65, 0xc800, v1
	v_add_u32_e32 v66, 0xca00, v1
	v_add_u32_e32 v67, 0xcc00, v1
	v_add_u32_e32 v68, 0xce00, v1
	s_waitcnt vmcnt(44) lgkmcnt(0)
	v_pk_fma_f32 v[44:45], v[44:45], v[48:49], v[50:51]
	ds_read_b128 v[48:51], v11 offset:160
	s_waitcnt vmcnt(42)
	v_pk_fma_f32 v[52:53], v[46:47], v[52:53], v[44:45]
	ds_read_b128 v[44:47], v11 offset:176
	s_waitcnt vmcnt(40) lgkmcnt(1)
	v_pk_fma_f32 v[48:49], v[48:49], v[54:55], v[52:53]
	s_waitcnt vmcnt(38)
	v_pk_fma_f32 v[24:25], v[50:51], v[24:25], v[48:49]
	v_or_b32_e32 v50, 0xb000, v1
	s_waitcnt vmcnt(36) lgkmcnt(0)
	v_pk_fma_f32 v[44:45], v[44:45], v[22:23], v[24:25]
	ds_read_b128 v[22:25], v11 offset:192
	s_waitcnt vmcnt(34)
	v_pk_fma_f32 v[20:21], v[46:47], v[20:21], v[44:45]
	ds_read_b128 v[44:47], v11 offset:208
	v_add_u32_e32 v51, 0xb200, v1
	v_add_u32_e32 v52, 0xac00, v1
	s_waitcnt vmcnt(32) lgkmcnt(1)
	v_pk_fma_f32 v[18:19], v[22:23], v[18:19], v[20:21]
	v_add_u32_e32 v22, 0xbc00, v1
	s_waitcnt vmcnt(30)
	v_pk_fma_f32 v[16:17], v[24:25], v[16:17], v[18:19]
	v_add_u32_e32 v23, 0xbe00, v1
	s_waitcnt vmcnt(28) lgkmcnt(0)
	v_pk_fma_f32 v[18:19], v[44:45], v[14:15], v[16:17]
	ds_read_b128 v[14:17], v11 offset:224
	s_waitcnt vmcnt(26)
	v_pk_fma_f32 v[12:13], v[46:47], v[12:13], v[18:19]
	ds_read_b128 v[44:47], v11 offset:240
	v_add_u32_e32 v53, 0xae00, v1
	v_add_u32_e32 v54, 0xb400, v1
	s_waitcnt vmcnt(24) lgkmcnt(1)
	v_pk_fma_f32 v[4:5], v[14:15], v[4:5], v[12:13]
	v_add_u32_e32 v55, 0xb600, v1
	s_waitcnt vmcnt(22)
	v_pk_fma_f32 v[48:49], v[16:17], v[2:3], v[4:5]
	ds_read_b128 v[2:5], v11 offset:256
	global_load_dword v16, v10, s[0:1]
	global_load_dword v17, v56, s[0:1]
	global_load_dword v14, v57, s[0:1]
	global_load_dword v15, v58, s[0:1]
	global_load_dword v12, v52, s[0:1]
	global_load_dword v13, v53, s[0:1]
	global_load_dword v20, v50, s[0:1]
	global_load_dword v21, v51, s[0:1]
	global_load_dword v18, v54, s[0:1]
	global_load_dword v19, v55, s[0:1]
	global_load_dword v24, v59, s[0:1]
	global_load_dword v25, v60, s[0:1]
	global_load_dword v22, v22, s[0:1]
	global_load_dword v23, v23, s[0:1]
	s_waitcnt vmcnt(34) lgkmcnt(1)
	v_pk_fma_f32 v[28:29], v[44:45], v[28:29], v[48:49]
	ds_read_b128 v[48:51], v11 offset:272
	s_waitcnt vmcnt(32)
	v_pk_fma_f32 v[8:9], v[46:47], v[8:9], v[28:29]
	ds_read_b128 v[44:47], v11 offset:288
	ds_read_b128 v[52:55], v11 offset:304
	s_waitcnt vmcnt(30) lgkmcnt(3)
	v_pk_fma_f32 v[2:3], v[2:3], v[34:35], v[8:9]
	v_or_b32_e32 v10, 0xd000, v1
	s_waitcnt vmcnt(28)
	v_pk_fma_f32 v[2:3], v[4:5], v[6:7], v[2:3]
	v_add_u32_e32 v34, 0xd200, v1
	s_waitcnt vmcnt(26) lgkmcnt(2)
	v_pk_fma_f32 v[28:29], v[48:49], v[32:33], v[2:3]
	ds_read_b128 v[6:9], v11 offset:320
	ds_read_b128 v[2:5], v11 offset:336
	s_waitcnt vmcnt(24)
	v_pk_fma_f32 v[26:27], v[50:51], v[26:27], v[28:29]
	v_add_u32_e32 v35, 0xd400, v1
	s_waitcnt vmcnt(22) lgkmcnt(3)
	v_pk_fma_f32 v[26:27], v[44:45], v[30:31], v[26:27]
	v_add_u32_e32 v48, 0xe800, v1
	s_waitcnt vmcnt(20)
	v_pk_fma_f32 v[26:27], v[46:47], v[36:37], v[26:27]
	v_add_u32_e32 v47, 0xe600, v1
	s_waitcnt vmcnt(18) lgkmcnt(2)
	v_pk_fma_f32 v[26:27], v[52:53], v[38:39], v[26:27]
	v_add_u32_e32 v39, 0xd600, v1
	s_waitcnt vmcnt(16)
	v_pk_fma_f32 v[26:27], v[54:55], v[40:41], v[26:27]
	v_add_u32_e32 v40, 0xd800, v1
	s_waitcnt vmcnt(14) lgkmcnt(1)
	v_pk_fma_f32 v[32:33], v[6:7], v[42:43], v[26:27]
	global_load_dword v26, v61, s[0:1]
	global_load_dword v27, v62, s[0:1]
	global_load_dword v6, v63, s[0:1]
	global_load_dword v7, v64, s[0:1]
	global_load_dword v30, v65, s[0:1]
	global_load_dword v31, v66, s[0:1]
	global_load_dword v28, v67, s[0:1]
	global_load_dword v29, v68, s[0:1]
	v_add_u32_e32 v41, 0xda00, v1
	v_add_u32_e32 v42, 0xdc00, v1
	v_add_u32_e32 v43, 0xde00, v1
	global_load_dword v36, v10, s[0:1]
	global_load_dword v37, v34, s[0:1]
	global_load_dword v38, v35, s[0:1]
	global_load_dword v39, v39, s[0:1]
	global_load_dword v40, v40, s[0:1]
	global_load_dword v41, v41, s[0:1]
	global_load_dword v42, v42, s[0:1]
	global_load_dword v43, v43, s[0:1]
	v_or_b32_e32 v10, 0xe000, v1
	v_add_u32_e32 v49, 0xea00, v1
	v_add_u32_e32 v50, 0xec00, v1
	v_add_u32_e32 v51, 0xee00, v1
	v_add_u32_e32 v34, 0xe200, v1
	v_add_u32_e32 v35, 0xe400, v1
	global_load_dword v44, v10, s[0:1]
	global_load_dword v45, v34, s[0:1]
	global_load_dword v46, v35, s[0:1]
	global_load_dword v47, v47, s[0:1]
	global_load_dword v48, v48, s[0:1]
	global_load_dword v49, v49, s[0:1]
	global_load_dword v50, v50, s[0:1]
	global_load_dword v51, v51, s[0:1]
	v_or_b32_e32 v10, 0xf000, v1
	v_add_u32_e32 v55, 0xf600, v1
	v_add_u32_e32 v56, 0xf800, v1
	v_add_u32_e32 v57, 0xfa00, v1
	v_add_u32_e32 v58, 0xfc00, v1
	v_add_u32_e32 v34, 0xf200, v1
	v_add_u32_e32 v35, 0xf400, v1
	v_add_u32_e32 v1, 0xfe00, v1
	global_load_dword v52, v10, s[0:1]
	global_load_dword v53, v34, s[0:1]
	global_load_dword v54, v35, s[0:1]
	global_load_dword v55, v55, s[0:1]
	global_load_dword v56, v56, s[0:1]
	global_load_dword v57, v57, s[0:1]
	global_load_dword v58, v58, s[0:1]
	global_load_dword v59, v1, s[0:1]
	ds_read_b32 v1, v11 offset:512
	s_waitcnt vmcnt(44)
	v_pk_fma_f32 v[8:9], v[8:9], v[16:17], v[32:33]
	ds_read_b128 v[32:35], v11 offset:352
	s_waitcnt vmcnt(42) lgkmcnt(2)
	v_pk_fma_f32 v[2:3], v[2:3], v[14:15], v[8:9]
	ds_read_b128 v[14:17], v11 offset:368
	s_waitcnt vmcnt(40)
	v_pk_fma_f32 v[2:3], v[4:5], v[12:13], v[2:3]
	s_waitcnt vmcnt(38) lgkmcnt(1)
	v_pk_fma_f32 v[8:9], v[32:33], v[20:21], v[2:3]
	ds_read_b128 v[2:5], v11 offset:384
	s_waitcnt vmcnt(36)
	v_pk_fma_f32 v[8:9], v[34:35], v[18:19], v[8:9]
	s_waitcnt vmcnt(34) lgkmcnt(1)
	v_pk_fma_f32 v[8:9], v[14:15], v[24:25], v[8:9]
	ds_read_b128 v[12:15], v11 offset:400
	s_waitcnt vmcnt(32)
	v_pk_fma_f32 v[8:9], v[16:17], v[22:23], v[8:9]
	ds_read_b128 v[16:19], v11 offset:416
	s_waitcnt vmcnt(30) lgkmcnt(2)
	v_pk_fma_f32 v[2:3], v[2:3], v[26:27], v[8:9]
	s_waitcnt vmcnt(28)
	v_pk_fma_f32 v[2:3], v[4:5], v[6:7], v[2:3]
	s_waitcnt vmcnt(26) lgkmcnt(1)
	v_pk_fma_f32 v[6:7], v[12:13], v[30:31], v[2:3]
	ds_read_b128 v[2:5], v11 offset:432
	s_waitcnt vmcnt(24)
	v_pk_fma_f32 v[6:7], v[14:15], v[28:29], v[6:7]
	s_waitcnt vmcnt(22) lgkmcnt(1)
	v_pk_fma_f32 v[12:13], v[16:17], v[36:37], v[6:7]
	ds_read_b128 v[6:9], v11 offset:448
	s_waitcnt vmcnt(20)
	v_pk_fma_f32 v[12:13], v[18:19], v[38:39], v[12:13]
	s_waitcnt vmcnt(18) lgkmcnt(1)
	v_pk_fma_f32 v[2:3], v[2:3], v[40:41], v[12:13]
	ds_read_b128 v[12:15], v11 offset:464
	s_waitcnt vmcnt(16)
	v_pk_fma_f32 v[2:3], v[4:5], v[42:43], v[2:3]
	s_waitcnt vmcnt(14) lgkmcnt(1)
	v_pk_fma_f32 v[6:7], v[6:7], v[44:45], v[2:3]
	ds_read_b128 v[2:5], v11 offset:480
	s_waitcnt vmcnt(12)
	v_pk_fma_f32 v[6:7], v[8:9], v[46:47], v[6:7]
	s_waitcnt vmcnt(10) lgkmcnt(1)
	v_pk_fma_f32 v[12:13], v[12:13], v[48:49], v[6:7]
	ds_read_b128 v[6:9], v11 offset:496
	s_waitcnt vmcnt(8)
	v_pk_fma_f32 v[12:13], v[14:15], v[50:51], v[12:13]
	s_waitcnt vmcnt(6) lgkmcnt(1)
	v_pk_fma_f32 v[2:3], v[2:3], v[52:53], v[12:13]
	s_waitcnt vmcnt(4)
	v_pk_fma_f32 v[2:3], v[4:5], v[54:55], v[2:3]
	s_waitcnt vmcnt(2) lgkmcnt(0)
	v_pk_fma_f32 v[2:3], v[6:7], v[56:57], v[2:3]
	s_waitcnt vmcnt(0)
	v_pk_fma_f32 v[2:3], v[8:9], v[58:59], v[2:3]
	s_nop 0
	v_add_f32_e32 v2, v2, v3
	v_mul_f32_e32 v1, v1, v2
.LBB1_5:                                ; %.sink.split
	v_lshl_add_u32 v2, s2, 7, v0
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[2:3], v[2:3], 2, s[4:5]
	global_store_dword v[2:3], v1, off
.LBB1_6:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z25tqm_dequantize_kernel_tq3PK16block_tq3_mi300xPKfPfi
		.amdhsa_group_segment_fixed_size 516
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 28
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 0
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 75
		.amdhsa_next_free_sgpr 10
		.amdhsa_accum_offset 76
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_tg_split 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end1:
	.size	_Z25tqm_dequantize_kernel_tq3PK16block_tq3_mi300xPKfPfi, .Lfunc_end1-_Z25tqm_dequantize_kernel_tq3PK16block_tq3_mi300xPKfPfi
                                        ; -- End function
	.set _Z25tqm_dequantize_kernel_tq3PK16block_tq3_mi300xPKfPfi.num_vgpr, 75
	.set _Z25tqm_dequantize_kernel_tq3PK16block_tq3_mi300xPKfPfi.num_agpr, 0
	.set _Z25tqm_dequantize_kernel_tq3PK16block_tq3_mi300xPKfPfi.numbered_sgpr, 10
	.set _Z25tqm_dequantize_kernel_tq3PK16block_tq3_mi300xPKfPfi.num_named_barrier, 0
	.set _Z25tqm_dequantize_kernel_tq3PK16block_tq3_mi300xPKfPfi.private_seg_size, 0
	.set _Z25tqm_dequantize_kernel_tq3PK16block_tq3_mi300xPKfPfi.uses_vcc, 1
	.set _Z25tqm_dequantize_kernel_tq3PK16block_tq3_mi300xPKfPfi.uses_flat_scratch, 0
	.set _Z25tqm_dequantize_kernel_tq3PK16block_tq3_mi300xPKfPfi.has_dyn_sized_stack, 0
	.set _Z25tqm_dequantize_kernel_tq3PK16block_tq3_mi300xPKfPfi.has_recursion, 0
	.set _Z25tqm_dequantize_kernel_tq3PK16block_tq3_mi300xPKfPfi.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 3444
; TotalNumSgprs: 16
; NumVgprs: 75
; NumAgprs: 0
; TotalNumVgprs: 75
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 516 bytes/workgroup (compile time only)
; SGPRBlocks: 1
; VGPRBlocks: 9
; NumSGPRsForWavesPerEU: 16
; NumVGPRsForWavesPerEU: 75
; AccumOffset: 76
; Occupancy: 6
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 18
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.protected	_Z24tqm_fused_dot_kernel_tq3PKfPK16block_tq3_mi300xPfii ; -- Begin function _Z24tqm_fused_dot_kernel_tq3PKfPK16block_tq3_mi300xPfii
	.globl	_Z24tqm_fused_dot_kernel_tq3PKfPK16block_tq3_mi300xPfii
	.p2align	8
	.type	_Z24tqm_fused_dot_kernel_tq3PKfPK16block_tq3_mi300xPfii,@function
_Z24tqm_fused_dot_kernel_tq3PKfPK16block_tq3_mi300xPfii: ; @_Z24tqm_fused_dot_kernel_tq3PKfPK16block_tq3_mi300xPfii
; %bb.0:
	s_load_dwordx2 s[6:7], s[0:1], 0x18
	s_waitcnt lgkmcnt(0)
	s_cmp_ge_i32 s2, s7
	s_cselect_b64 s[4:5], -1, 0
	s_cmp_ge_i32 s3, s6
	s_cselect_b64 s[8:9], -1, 0
	s_or_b64 s[4:5], s[8:9], s[4:5]
	s_and_b64 vcc, exec, s[4:5]
	s_cbranch_vccnz .LBB2_13
; %bb.1:
	s_load_dwordx2 s[4:5], s[0:1], 0x8
	s_mul_i32 s8, s2, 52
	s_mul_hi_i32 s6, s2, 52
	s_waitcnt lgkmcnt(0)
	s_add_u32 s8, s4, s8
	s_addc_u32 s9, s5, s6
	v_cmp_eq_u32_e64 s[4:5], 0, v0
	s_and_saveexec_b64 s[10:11], s[4:5]
	s_cbranch_execz .LBB2_3
; %bb.2:
	v_mov_b32_e32 v1, 0
	global_load_dword v2, v1, s[8:9]
	s_waitcnt vmcnt(0)
	ds_write_b32 v1, v2 offset:8
.LBB2_3:
	s_or_b64 exec, exec, s[10:11]
	v_mov_b32_e32 v3, 0
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b32 v1, v3 offset:8
	s_mov_b32 s6, 0x26901d7d
	s_waitcnt lgkmcnt(0)
	v_cmp_lt_f32_e64 s[10:11], |v1|, s6
	s_and_b64 vcc, exec, s[10:11]
	s_cbranch_vccnz .LBB2_9
; %bb.4:                                ; %.preheader
	v_lshrrev_b32_e32 v1, 6, v0
	v_lshlrev_b32_e32 v2, 3, v1
	global_load_dwordx2 v[4:5], v2, s[8:9] offset:4
	global_load_dwordx2 v[6:7], v2, s[8:9] offset:20
	global_load_dwordx2 v[8:9], v2, s[8:9] offset:36
	s_load_dwordx2 s[8:9], s[0:1], 0x0
	s_getpc_b64 s[10:11]
	s_add_u32 s10, s10, d_cb3@rel32@lo+4
	s_addc_u32 s11, s11, d_cb3@rel32@hi+12
	v_lshl_add_u32 v10, s3, 7, v0
	v_ashrrev_i32_e32 v11, 31, v10
	s_waitcnt lgkmcnt(0)
	v_lshl_add_u64 v[10:11], v[10:11], 2, s[8:9]
	global_load_dword v10, v[10:11], off
	s_waitcnt vmcnt(3)
	v_lshrrev_b64 v[4:5], v0, v[4:5]
	s_waitcnt vmcnt(2)
	v_lshrrev_b64 v[6:7], v0, v[6:7]
	v_and_b32_e32 v2, 1, v4
	s_waitcnt vmcnt(1)
	v_lshrrev_b64 v[8:9], v0, v[8:9]
	v_lshlrev_b32_e32 v6, 3, v6
	v_lshlrev_b32_e32 v2, 2, v2
	v_lshlrev_b32_e32 v7, 4, v8
	v_lshl_add_u64 v[4:5], s[10:11], 0, v[2:3]
	v_and_b32_e32 v2, 8, v6
	v_lshl_add_u64 v[4:5], v[4:5], 0, v[2:3]
	v_and_b32_e32 v2, 16, v7
	v_lshl_add_u64 v[2:3], v[4:5], 0, v[2:3]
	global_load_dword v2, v[2:3], off
	v_mbcnt_lo_u32_b32 v3, -1, 0
	v_mov_b32_e32 v4, 0x80
	v_mbcnt_hi_u32_b32 v3, -1, v3
	v_lshl_or_b32 v4, v3, 2, v4
	v_and_b32_e32 v0, 63, v0
	s_waitcnt vmcnt(0)
	v_mul_f32_e32 v5, v10, v2
	ds_bpermute_b32 v4, v4, v5
	v_and_b32_e32 v5, 63, v3
	v_cmp_gt_u32_e32 vcc, 48, v5
	s_waitcnt lgkmcnt(0)
	v_fmac_f32_e32 v4, v10, v2
	v_cndmask_b32_e64 v6, 0, 16, vcc
	v_add_lshl_u32 v6, v6, v3, 2
	ds_bpermute_b32 v2, v6, v4
	v_cmp_gt_u32_e32 vcc, 56, v5
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v2, v4, v2
	v_cndmask_b32_e64 v6, 0, 8, vcc
	v_add_lshl_u32 v6, v6, v3, 2
	ds_bpermute_b32 v4, v6, v2
	v_cmp_gt_u32_e32 vcc, 60, v5
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v2, v2, v4
	v_cndmask_b32_e64 v6, 0, 4, vcc
	v_add_lshl_u32 v6, v6, v3, 2
	ds_bpermute_b32 v4, v6, v2
	v_cmp_gt_u32_e32 vcc, 62, v5
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v2, v2, v4
	v_cndmask_b32_e64 v6, 0, 2, vcc
	v_add_lshl_u32 v6, v6, v3, 2
	ds_bpermute_b32 v4, v6, v2
	v_cmp_ne_u32_e32 vcc, 63, v5
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v2, v2, v4
	v_addc_co_u32_e32 v3, vcc, 0, v3, vcc
	v_lshlrev_b32_e32 v3, 2, v3
	ds_bpermute_b32 v3, v3, v2
	v_cmp_eq_u32_e32 vcc, 0, v0
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB2_6
; %bb.5:
	v_lshlrev_b32_e32 v0, 2, v1
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v1, v2, v3
	ds_write_b32 v0, v1
.LBB2_6:
	s_or_b64 exec, exec, s[8:9]
	s_mov_b64 s[10:11], 0
	s_mov_b64 s[8:9], 0
	s_waitcnt lgkmcnt(0)
	s_barrier
                                        ; implicit-def: $vgpr0
	s_and_saveexec_b64 s[12:13], s[4:5]
	s_xor_b64 s[12:13], exec, s[12:13]
	s_cbranch_execz .LBB2_8
; %bb.7:
	v_mov_b32_e32 v2, 0
	ds_read_b64 v[0:1], v2
	ds_read_b32 v2, v2 offset:8
	s_mov_b64 s[8:9], exec
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v0, v1, v0
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v0, v0, v2
.LBB2_8:                                ; %Flow63
	s_or_b64 exec, exec, s[12:13]
	s_and_b64 vcc, exec, s[10:11]
	s_cbranch_vccnz .LBB2_10
	s_branch .LBB2_11
.LBB2_9:
	s_mov_b64 s[8:9], 0
                                        ; implicit-def: $vgpr0
	s_cbranch_execz .LBB2_11
.LBB2_10:
	s_andn2_b64 s[8:9], s[8:9], exec
	s_and_b64 s[4:5], s[4:5], exec
	v_mov_b32_e32 v0, 0
	s_or_b64 s[8:9], s[8:9], s[4:5]
.LBB2_11:                               ; %Flow64
	s_and_saveexec_b64 s[4:5], s[8:9]
	s_cbranch_execz .LBB2_13
; %bb.12:                               ; %.sink.split
	s_load_dwordx2 s[0:1], s[0:1], 0x10
	s_mul_i32 s3, s7, s3
	s_add_i32 s2, s3, s2
	s_ashr_i32 s3, s2, 31
	s_lshl_b64 s[2:3], s[2:3], 2
	s_waitcnt lgkmcnt(0)
	s_add_u32 s0, s0, s2
	s_addc_u32 s1, s1, s3
	v_mov_b32_e32 v1, 0
	global_store_dword v1, v0, s[0:1]
.LBB2_13:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z24tqm_fused_dot_kernel_tq3PKfPK16block_tq3_mi300xPfii
		.amdhsa_group_segment_fixed_size 12
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 32
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 0
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 12
		.amdhsa_next_free_sgpr 14
		.amdhsa_accum_offset 12
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_tg_split 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end2:
	.size	_Z24tqm_fused_dot_kernel_tq3PKfPK16block_tq3_mi300xPfii, .Lfunc_end2-_Z24tqm_fused_dot_kernel_tq3PKfPK16block_tq3_mi300xPfii
                                        ; -- End function
	.set _Z24tqm_fused_dot_kernel_tq3PKfPK16block_tq3_mi300xPfii.num_vgpr, 12
	.set _Z24tqm_fused_dot_kernel_tq3PKfPK16block_tq3_mi300xPfii.num_agpr, 0
	.set _Z24tqm_fused_dot_kernel_tq3PKfPK16block_tq3_mi300xPfii.numbered_sgpr, 14
	.set _Z24tqm_fused_dot_kernel_tq3PKfPK16block_tq3_mi300xPfii.num_named_barrier, 0
	.set _Z24tqm_fused_dot_kernel_tq3PKfPK16block_tq3_mi300xPfii.private_seg_size, 0
	.set _Z24tqm_fused_dot_kernel_tq3PKfPK16block_tq3_mi300xPfii.uses_vcc, 1
	.set _Z24tqm_fused_dot_kernel_tq3PKfPK16block_tq3_mi300xPfii.uses_flat_scratch, 0
	.set _Z24tqm_fused_dot_kernel_tq3PKfPK16block_tq3_mi300xPfii.has_dyn_sized_stack, 0
	.set _Z24tqm_fused_dot_kernel_tq3PKfPK16block_tq3_mi300xPfii.has_recursion, 0
	.set _Z24tqm_fused_dot_kernel_tq3PKfPK16block_tq3_mi300xPfii.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 776
; TotalNumSgprs: 20
; NumVgprs: 12
; NumAgprs: 0
; TotalNumVgprs: 12
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 12 bytes/workgroup (compile time only)
; SGPRBlocks: 2
; VGPRBlocks: 1
; NumSGPRsForWavesPerEU: 20
; NumVGPRsForWavesPerEU: 12
; AccumOffset: 12
; Occupancy: 8
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 2
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.protected	_Z23tqm_quantize_kernel_tq4PKfS0_P16block_tq4_mi300xi ; -- Begin function _Z23tqm_quantize_kernel_tq4PKfS0_P16block_tq4_mi300xi
	.globl	_Z23tqm_quantize_kernel_tq4PKfS0_P16block_tq4_mi300xi
	.p2align	8
	.type	_Z23tqm_quantize_kernel_tq4PKfS0_P16block_tq4_mi300xi,@function
_Z23tqm_quantize_kernel_tq4PKfS0_P16block_tq4_mi300xi: ; @_Z23tqm_quantize_kernel_tq4PKfS0_P16block_tq4_mi300xi
; %bb.0:
	s_load_dword s3, s[0:1], 0x18
	s_waitcnt lgkmcnt(0)
	s_cmp_ge_i32 s2, s3
	s_cbranch_scc1 .LBB3_22
; %bb.1:
	s_load_dwordx2 s[4:5], s[0:1], 0x0
	s_load_dwordx2 s[8:9], s[0:1], 0x10
	v_lshl_add_u32 v2, s2, 7, v0
	v_ashrrev_i32_e32 v3, 31, v2
	s_waitcnt lgkmcnt(0)
	v_lshl_add_u64 v[2:3], v[2:3], 2, s[4:5]
	global_load_dword v1, v[2:3], off
	v_lshlrev_b32_e32 v2, 2, v0
	v_mov_b32_e32 v3, 0x80
	s_waitcnt vmcnt(0)
	ds_write_b32 v2, v1
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b32 v1, v2
	v_mbcnt_lo_u32_b32 v2, -1, 0
	v_mbcnt_hi_u32_b32 v2, -1, v2
	v_lshl_or_b32 v3, v2, 2, v3
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v4, v1, v1
	ds_bpermute_b32 v3, v3, v4
	v_and_b32_e32 v4, 63, v2
	v_cmp_gt_u32_e32 vcc, 48, v4
	s_waitcnt lgkmcnt(0)
	v_fmac_f32_e32 v3, v1, v1
	v_cndmask_b32_e64 v5, 0, 16, vcc
	v_add_lshl_u32 v5, v5, v2, 2
	ds_bpermute_b32 v1, v5, v3
	v_cmp_gt_u32_e32 vcc, 56, v4
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v1, v3, v1
	v_cndmask_b32_e64 v5, 0, 8, vcc
	v_add_lshl_u32 v5, v5, v2, 2
	ds_bpermute_b32 v3, v5, v1
	v_cmp_gt_u32_e32 vcc, 60, v4
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v1, v1, v3
	v_cndmask_b32_e64 v5, 0, 4, vcc
	v_add_lshl_u32 v5, v5, v2, 2
	ds_bpermute_b32 v3, v5, v1
	v_cmp_gt_u32_e32 vcc, 62, v4
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v1, v1, v3
	v_cndmask_b32_e64 v5, 0, 2, vcc
	v_add_lshl_u32 v5, v5, v2, 2
	ds_bpermute_b32 v3, v5, v1
	v_cmp_ne_u32_e32 vcc, 63, v4
	s_nop 1
	v_addc_co_u32_e32 v4, vcc, 0, v2, vcc
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v2, v1, v3
	v_lshlrev_b32_e32 v1, 2, v4
	ds_bpermute_b32 v3, v1, v2
	v_and_b32_e32 v4, 63, v0
	v_lshrrev_b32_e32 v1, 6, v0
	v_cmp_eq_u32_e64 s[6:7], 0, v4
	s_and_saveexec_b64 s[4:5], s[6:7]
	s_cbranch_execz .LBB3_3
; %bb.2:
	v_lshlrev_b32_e32 v4, 2, v1
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v2, v2, v3
	ds_write_b32 v4, v2 offset:512
.LBB3_3:
	s_or_b64 exec, exec, s[4:5]
	v_mov_b32_e32 v2, 0
	v_cmp_eq_u32_e64 s[4:5], 0, v0
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_and_saveexec_b64 s[10:11], s[4:5]
	s_cbranch_execz .LBB3_5
; %bb.4:
	ds_read_b64 v[4:5], v2 offset:512
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v3, v5, v4
	ds_write_b32 v2, v3 offset:520
.LBB3_5:
	s_or_b64 exec, exec, s[10:11]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b32 v2, v2 offset:520
	s_mul_hi_i32 s3, s2, 0x44
	s_mulk_i32 s2, 0x44
	s_add_u32 s2, s8, s2
	s_mov_b32 s8, 0x26901d7d
	s_waitcnt lgkmcnt(0)
	v_sqrt_f32_e32 v126, v2
	s_addc_u32 s3, s9, s3
	v_cmp_le_f32_e32 vcc, s8, v126
	s_mov_b64 s[8:9], -1
	s_cbranch_vccz .LBB3_17
; %bb.6:
	s_load_dwordx2 s[0:1], s[0:1], 0x8
	v_lshlrev_b32_e32 v6, 9, v0
	v_mov_b32_e32 v127, 0
	s_waitcnt lgkmcnt(0)
	global_load_dwordx4 v[10:13], v6, s[0:1] offset:48
	global_load_dwordx4 v[2:5], v6, s[0:1] offset:32
	s_waitcnt vmcnt(0)
	scratch_store_dwordx4 off, v[2:5], off offset:80 ; 16-byte Folded Spill
	global_load_dwordx4 v[2:5], v6, s[0:1] offset:16
	s_waitcnt vmcnt(0)
	scratch_store_dwordx4 off, v[2:5], off offset:208 ; 16-byte Folded Spill
	global_load_dwordx4 v[2:5], v6, s[0:1]
	s_waitcnt vmcnt(0)
	scratch_store_dwordx4 off, v[2:5], off offset:96 ; 16-byte Folded Spill
	ds_read_b128 v[2:5], v127
	s_waitcnt lgkmcnt(0)
	scratch_store_dwordx4 off, v[2:5], off offset:112 ; 16-byte Folded Spill
	ds_read_b128 v[42:45], v127 offset:16
	ds_read_b128 v[2:5], v127 offset:32
	ds_read_b128 v[122:125], v127 offset:48
	s_waitcnt lgkmcnt(1)
	scratch_store_dwordx4 off, v[2:5], off offset:544 ; 16-byte Folded Spill
	global_load_dwordx4 v[32:35], v6, s[0:1] offset:112
	global_load_dwordx4 v[2:5], v6, s[0:1] offset:96
	s_waitcnt vmcnt(0)
	scratch_store_dwordx4 off, v[2:5], off offset:128 ; 16-byte Folded Spill
	global_load_dwordx4 v[54:57], v6, s[0:1] offset:80
	global_load_dwordx4 v[2:5], v6, s[0:1] offset:64
	s_waitcnt vmcnt(0)
	scratch_store_dwordx4 off, v[2:5], off offset:464 ; 16-byte Folded Spill
	ds_read_b128 v[62:65], v127 offset:80
	ds_read_b128 v[2:5], v127 offset:64
	s_waitcnt lgkmcnt(0)
	scratch_store_dwordx4 off, v[2:5], off offset:512 ; 16-byte Folded Spill
	ds_read_b128 v[58:61], v127 offset:112
	ds_read_b128 v[2:5], v127 offset:96
	s_waitcnt lgkmcnt(0)
	scratch_store_dwordx4 off, v[2:5], off offset:144 ; 16-byte Folded Spill
	global_load_dwordx4 v[2:5], v6, s[0:1] offset:176
	s_waitcnt vmcnt(0)
	scratch_store_dwordx4 off, v[2:5], off offset:592 ; 16-byte Folded Spill
	global_load_dwordx4 v[2:5], v6, s[0:1] offset:160
	s_waitcnt vmcnt(0)
	scratch_store_dwordx4 off, v[2:5], off offset:160 ; 16-byte Folded Spill
	global_load_dwordx4 v[70:73], v6, s[0:1] offset:144
	global_load_dwordx4 v[2:5], v6, s[0:1] offset:128
	s_waitcnt vmcnt(0)
	scratch_store_dwordx4 off, v[2:5], off offset:576 ; 16-byte Folded Spill
	ds_read_b128 v[104:107], v127 offset:144
	ds_read_b128 v[2:5], v127 offset:128
	s_waitcnt lgkmcnt(0)
	scratch_store_dwordx4 off, v[2:5], off offset:608 ; 16-byte Folded Spill
	ds_read_b128 v[2:5], v127 offset:176
	s_waitcnt lgkmcnt(0)
	scratch_store_dwordx4 off, v[2:5], off offset:416 ; 16-byte Folded Spill
	ds_read_b128 v[2:5], v127 offset:160
	s_waitcnt lgkmcnt(0)
	scratch_store_dwordx4 off, v[2:5], off offset:176 ; 16-byte Folded Spill
	global_load_dwordx4 v[86:89], v6, s[0:1] offset:240
	global_load_dwordx4 v[2:5], v6, s[0:1] offset:224
	s_waitcnt vmcnt(0)
	scratch_store_dwordx4 off, v[2:5], off  ; 16-byte Folded Spill
	global_load_dwordx4 v[26:29], v6, s[0:1] offset:208
	global_load_dwordx4 v[2:5], v6, s[0:1] offset:192
	s_waitcnt vmcnt(0)
	scratch_store_dwordx4 off, v[2:5], off offset:624 ; 16-byte Folded Spill
	ds_read_b128 v[96:99], v127 offset:208
	ds_read_b128 v[2:5], v127 offset:192
	s_waitcnt lgkmcnt(0)
	scratch_store_dwordx4 off, v[2:5], off offset:480 ; 16-byte Folded Spill
	ds_read_b128 v[2:5], v127 offset:240
	s_waitcnt lgkmcnt(0)
	scratch_store_dwordx4 off, v[2:5], off offset:432 ; 16-byte Folded Spill
	ds_read_b128 v[2:5], v127 offset:224
	s_waitcnt lgkmcnt(0)
	scratch_store_dwordx4 off, v[2:5], off offset:192 ; 16-byte Folded Spill
	global_load_dwordx4 v[100:103], v6, s[0:1] offset:304
	global_load_dwordx4 v[2:5], v6, s[0:1] offset:288
	s_waitcnt vmcnt(0)
	scratch_store_dwordx4 off, v[2:5], off offset:16 ; 16-byte Folded Spill
	global_load_dwordx4 v[66:69], v6, s[0:1] offset:272
	global_load_dwordx4 v[2:5], v6, s[0:1] offset:256
	s_waitcnt vmcnt(0)
	scratch_store_dwordx4 off, v[2:5], off offset:32 ; 16-byte Folded Spill
	ds_read_b128 v[90:93], v127 offset:272
	ds_read_b128 v[2:5], v127 offset:256
	s_waitcnt lgkmcnt(0)
	scratch_store_dwordx4 off, v[2:5], off offset:48 ; 16-byte Folded Spill
	ds_read_b128 v[82:85], v127 offset:304
	ds_read_b128 v[2:5], v127 offset:288
	s_waitcnt lgkmcnt(0)
	scratch_store_dwordx4 off, v[2:5], off offset:224 ; 16-byte Folded Spill
	global_load_dwordx4 v[50:53], v6, s[0:1] offset:368
	global_load_dwordx4 v[2:5], v6, s[0:1] offset:352
	s_waitcnt vmcnt(0)
	scratch_store_dwordx4 off, v[2:5], off offset:352 ; 16-byte Folded Spill
	global_load_dwordx4 v[78:81], v6, s[0:1] offset:336
	global_load_dwordx4 v[2:5], v6, s[0:1] offset:320
	s_waitcnt vmcnt(0)
	scratch_store_dwordx4 off, v[2:5], off offset:384 ; 16-byte Folded Spill
	ds_read_b128 v[36:39], v127 offset:336
	ds_read_b128 v[2:5], v127 offset:320
	s_waitcnt lgkmcnt(0)
	scratch_store_dwordx4 off, v[2:5], off offset:64 ; 16-byte Folded Spill
	ds_read_b128 v[108:111], v127 offset:368
	ds_read_b128 v[2:5], v127 offset:352
	s_waitcnt lgkmcnt(0)
	scratch_store_dwordx4 off, v[2:5], off offset:400 ; 16-byte Folded Spill
	global_load_dwordx4 v[2:5], v6, s[0:1] offset:432
	s_waitcnt vmcnt(0)
	scratch_store_dwordx4 off, v[2:5], off offset:448 ; 16-byte Folded Spill
	global_load_dwordx4 v[2:5], v6, s[0:1] offset:416
	s_waitcnt vmcnt(0)
	scratch_store_dwordx4 off, v[2:5], off offset:256 ; 16-byte Folded Spill
	global_load_dwordx4 v[14:17], v6, s[0:1] offset:400
	global_load_dwordx4 v[2:5], v6, s[0:1] offset:384
	s_waitcnt vmcnt(0)
	scratch_store_dwordx4 off, v[2:5], off offset:272 ; 16-byte Folded Spill
	ds_read_b128 v[46:49], v127 offset:400
	ds_read_b128 v[2:5], v127 offset:384
	s_waitcnt lgkmcnt(0)
	scratch_store_dwordx4 off, v[2:5], off offset:368 ; 16-byte Folded Spill
	ds_read_b128 v[2:5], v127 offset:432
	s_waitcnt lgkmcnt(0)
	scratch_store_dwordx4 off, v[2:5], off offset:528 ; 16-byte Folded Spill
	ds_read_b128 v[2:5], v127 offset:416
	s_waitcnt lgkmcnt(0)
	scratch_store_dwordx4 off, v[2:5], off offset:336 ; 16-byte Folded Spill
	global_load_dwordx4 v[2:5], v6, s[0:1] offset:496
	s_waitcnt vmcnt(0)
	scratch_store_dwordx4 off, v[2:5], off offset:496 ; 16-byte Folded Spill
	global_load_dwordx4 v[2:5], v6, s[0:1] offset:480
	s_waitcnt vmcnt(0)
	scratch_store_dwordx4 off, v[2:5], off offset:240 ; 16-byte Folded Spill
	global_load_dwordx4 v[20:23], v6, s[0:1] offset:464
	global_load_dwordx4 v[2:5], v6, s[0:1] offset:448
	s_getpc_b64 s[0:1]
	s_add_u32 s0, s0, d_cb4@rel32@lo+4
	s_addc_u32 s1, s1, d_cb4@rel32@hi+12
	s_load_dwordx16 s[8:23], s[0:1], 0x0
	s_waitcnt vmcnt(0)
	scratch_store_dwordx4 off, v[2:5], off offset:288 ; 16-byte Folded Spill
	ds_read_b128 v[74:77], v127 offset:464
	ds_read_b128 v[2:5], v127 offset:448
	s_waitcnt lgkmcnt(0)
	scratch_store_dwordx4 off, v[2:5], off offset:320 ; 16-byte Folded Spill
	ds_read_b128 v[2:5], v127 offset:496
	s_waitcnt lgkmcnt(0)
	scratch_store_dwordx4 off, v[2:5], off offset:560 ; 16-byte Folded Spill
	ds_read_b128 v[2:5], v127 offset:480
	s_waitcnt lgkmcnt(0)
	scratch_store_dwordx4 off, v[2:5], off offset:304 ; 16-byte Folded Spill
	s_and_saveexec_b64 s[0:1], s[4:5]
	s_cbranch_execz .LBB3_8
; %bb.7:
	global_store_dword v127, v126, s[2:3]
.LBB3_8:
	s_or_b64 exec, exec, s[0:1]
	scratch_load_dwordx4 v[114:117], off, off offset:208 ; 16-byte Folded Reload
	v_pk_mul_f32 v[18:19], v[122:123], v[10:11]
	v_mov_b64_e32 v[6:7], v[104:105]
	v_mov_b64_e32 v[8:9], v[106:107]
	v_mov_b64_e32 v[122:123], v[12:13]
	s_waitcnt vmcnt(0)
	v_pk_fma_f32 v[10:11], v[42:43], v[114:115], v[18:19]
	s_nop 0
	v_pk_fma_f32 v[10:11], v[62:63], v[54:55], v[10:11]
	v_mov_b64_e32 v[114:115], v[98:99]
	v_pk_fma_f32 v[10:11], v[58:59], v[32:33], v[10:11]
	scratch_load_dwordx4 v[104:107], off, off offset:416 ; 16-byte Folded Reload
	scratch_load_dwordx4 v[30:33], off, off offset:592 ; 16-byte Folded Reload
	v_mov_b64_e32 v[112:113], v[96:97]
	scratch_load_dwordx4 v[94:97], off, off offset:432 ; 16-byte Folded Reload
	v_pk_fma_f32 v[10:11], v[6:7], v[70:71], v[10:11]
	v_mov_b64_e32 v[70:71], v[68:69]
	v_mov_b64_e32 v[68:69], v[66:67]
	v_mov_b64_e32 v[62:63], v[38:39]
	s_waitcnt vmcnt(1)
	v_pk_fma_f32 v[10:11], v[104:105], v[30:31], v[10:11]
	v_mov_b64_e32 v[30:31], v[28:29]
	v_mov_b64_e32 v[28:29], v[26:27]
	v_pk_fma_f32 v[10:11], v[112:113], v[28:29], v[10:11]
	scratch_load_dwordx4 v[26:29], off, off offset:528 ; 16-byte Folded Reload
	s_waitcnt vmcnt(1)
	v_pk_fma_f32 v[10:11], v[94:95], v[86:87], v[10:11]
	v_mov_b64_e32 v[94:95], v[92:93]
	v_mov_b64_e32 v[92:93], v[90:91]
	v_mov_b64_e32 v[86:87], v[84:85]
	v_pk_fma_f32 v[10:11], v[92:93], v[68:69], v[10:11]
	v_mov_b64_e32 v[84:85], v[82:83]
	v_mov_b64_e32 v[82:83], v[80:81]
	v_pk_fma_f32 v[10:11], v[84:85], v[100:101], v[10:11]
	v_mov_b64_e32 v[80:81], v[78:79]
	v_pk_fma_f32 v[10:11], v[36:37], v[80:81], v[10:11]
	v_mov_b64_e32 v[68:69], v[22:23]
	v_pk_fma_f32 v[10:11], v[108:109], v[50:51], v[10:11]
	v_mov_b64_e32 v[108:109], v[48:49]
	v_pk_fma_f32 v[10:11], v[46:47], v[14:15], v[10:11]
	scratch_load_dwordx4 v[46:49], off, off offset:448 ; 16-byte Folded Reload
	v_mov_b64_e32 v[66:67], v[20:21]
	scratch_load_dwordx4 v[78:81], off, off offset:496 ; 16-byte Folded Reload
	scratch_load_dwordx4 v[22:25], off, off offset:560 ; 16-byte Folded Reload
	scratch_load_dwordx4 v[90:93], off, off offset:80 ; 16-byte Folded Reload
	scratch_load_dwordx4 v[4:7], off, off offset:544 ; 16-byte Folded Reload
	scratch_load_dwordx4 v[40:43], off, off offset:96 ; 16-byte Folded Reload
	scratch_load_dwordx4 v[18:21], off, off offset:112 ; 16-byte Folded Reload
	scratch_load_dwordx4 v[98:101], off, off offset:128 ; 16-byte Folded Reload
	v_mov_b64_e32 v[104:105], v[16:17]
	v_mov_b64_e32 v[58:59], v[82:83]
	scratch_load_dwordx4 v[36:39], off, off offset:512 ; 16-byte Folded Reload
	s_waitcnt vmcnt(8)
	v_pk_fma_f32 v[10:11], v[26:27], v[46:47], v[10:11]
	s_nop 0
	v_pk_fma_f32 v[2:3], v[74:75], v[66:67], v[10:11]
	v_mov_b64_e32 v[66:67], v[32:33]
	s_waitcnt vmcnt(6)
	v_pk_fma_f32 v[2:3], v[22:23], v[78:79], v[2:3]
	s_waitcnt vmcnt(4)
	v_mov_b64_e32 v[50:51], v[6:7]
	v_pk_fma_f32 v[2:3], v[4:5], v[90:91], v[2:3]
	scratch_load_dwordx4 v[4:7], off, off offset:464 ; 16-byte Folded Reload
	s_waitcnt vmcnt(3)
	v_pk_fma_f32 v[2:3], v[18:19], v[40:41], v[2:3]
	v_mov_b64_e32 v[74:75], v[30:31]
	s_waitcnt vmcnt(1)
	v_mov_b64_e32 v[40:41], v[38:39]
	s_waitcnt vmcnt(0)
	v_pk_fma_f32 v[2:3], v[36:37], v[4:5], v[2:3]
	scratch_load_dwordx4 v[36:39], off, off offset:144 ; 16-byte Folded Reload
	v_mov_b64_e32 v[54:55], v[6:7]
	scratch_load_dwordx4 v[4:7], off, off offset:576 ; 16-byte Folded Reload
	scratch_load_dwordx4 v[16:19], off, off offset:608 ; 16-byte Folded Reload
	scratch_load_dwordx4 v[118:121], off, off offset:160 ; 16-byte Folded Reload
	s_waitcnt vmcnt(3)
	v_pk_fma_f32 v[2:3], v[36:37], v[98:99], v[2:3]
	s_waitcnt vmcnt(2)
	v_mov_b64_e32 v[46:47], v[6:7]
	s_waitcnt vmcnt(1)
	v_pk_fma_f32 v[2:3], v[16:17], v[4:5], v[2:3]
	v_mov_b64_e32 v[22:23], v[18:19]
	scratch_load_dwordx4 v[16:19], off, off offset:176 ; 16-byte Folded Reload
	scratch_load_dwordx4 v[4:7], off, off offset:480 ; 16-byte Folded Reload
	scratch_load_dwordx4 v[82:85], off, off offset:624 ; 16-byte Folded Reload
	s_waitcnt vmcnt(1)
	v_mov_b64_e32 v[36:37], v[6:7]
	v_pk_fma_f32 v[2:3], v[16:17], v[118:119], v[2:3]
	s_waitcnt vmcnt(0)
	v_mov_b64_e32 v[26:27], v[84:85]
	v_pk_fma_f32 v[2:3], v[4:5], v[82:83], v[2:3]
	scratch_load_dwordx4 v[82:85], off, off offset:192 ; 16-byte Folded Reload
	scratch_load_dwordx4 v[4:7], off, off   ; 16-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_pk_fma_f32 v[2:3], v[82:83], v[4:5], v[2:3]
	scratch_load_dwordx4 v[4:7], off, off offset:32 ; 16-byte Folded Reload
	scratch_load_dwordx4 v[10:13], off, off offset:48 ; 16-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_pk_fma_f32 v[2:3], v[10:11], v[4:5], v[2:3]
	scratch_load_dwordx4 v[30:33], off, off offset:224 ; 16-byte Folded Reload
	scratch_load_dwordx4 v[4:7], off, off offset:16 ; 16-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_pk_fma_f32 v[2:3], v[30:31], v[4:5], v[2:3]
	scratch_load_dwordx4 v[4:7], off, off offset:384 ; 16-byte Folded Reload
	scratch_load_dwordx4 v[10:13], off, off offset:64 ; 16-byte Folded Reload
	s_waitcnt vmcnt(1)
	v_mov_b64_e32 v[118:119], v[6:7]
	s_waitcnt vmcnt(0)
	v_pk_fma_f32 v[2:3], v[10:11], v[4:5], v[2:3]
	scratch_load_dwordx4 v[4:7], off, off offset:352 ; 16-byte Folded Reload
	scratch_load_dwordx4 v[10:13], off, off offset:400 ; 16-byte Folded Reload
	s_waitcnt vmcnt(1)
	v_mov_b64_e32 v[98:99], v[6:7]
	s_waitcnt vmcnt(0)
	v_pk_fma_f32 v[2:3], v[10:11], v[4:5], v[2:3]
	v_mov_b64_e32 v[112:113], v[12:13]
	scratch_load_dwordx4 v[4:7], off, off offset:272 ; 16-byte Folded Reload
	scratch_load_dwordx4 v[10:13], off, off offset:368 ; 16-byte Folded Reload
	s_waitcnt vmcnt(1)
	v_mov_b64_e32 v[82:83], v[6:7]
	s_waitcnt vmcnt(0)
	v_pk_fma_f32 v[2:3], v[10:11], v[4:5], v[2:3]
	v_mov_b64_e32 v[90:91], v[12:13]
	scratch_load_dwordx4 v[4:7], off, off offset:256 ; 16-byte Folded Reload
	scratch_load_dwordx4 v[10:13], off, off offset:336 ; 16-byte Folded Reload
	s_waitcnt vmcnt(1)
	v_mov_b64_e32 v[30:31], v[6:7]
	s_waitcnt vmcnt(0)
	v_pk_fma_f32 v[2:3], v[10:11], v[4:5], v[2:3]
	v_mov_b64_e32 v[78:79], v[12:13]
	scratch_load_dwordx4 v[4:7], off, off offset:288 ; 16-byte Folded Reload
	scratch_load_dwordx4 v[12:15], off, off offset:320 ; 16-byte Folded Reload
	s_waitcnt vmcnt(1)
	v_mov_b64_e32 v[16:17], v[6:7]
	s_waitcnt vmcnt(0)
	v_pk_fma_f32 v[2:3], v[12:13], v[4:5], v[2:3]
	scratch_load_dwordx4 v[4:7], off, off offset:240 ; 16-byte Folded Reload
	scratch_load_dwordx4 v[10:13], off, off offset:304 ; 16-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_pk_fma_f32 v[2:3], v[10:11], v[4:5], v[2:3]
	s_nop 0
	v_pk_fma_f32 v[2:3], v[124:125], v[122:123], v[2:3]
	v_rcp_f32_e32 v4, v126
	v_pk_fma_f32 v[2:3], v[44:45], v[116:117], v[2:3]
	v_lshlrev_b32_e32 v126, 3, v1
	v_pk_fma_f32 v[2:3], v[64:65], v[56:57], v[2:3]
	s_nop 0
	v_pk_fma_f32 v[2:3], v[60:61], v[34:35], v[2:3]
	s_nop 0
	v_pk_fma_f32 v[2:3], v[8:9], v[72:73], v[2:3]
	scratch_load_dwordx4 v[8:11], off, off  ; 16-byte Folded Reload
	v_pk_fma_f32 v[2:3], v[106:107], v[66:67], v[2:3]
	s_nop 0
	v_pk_fma_f32 v[2:3], v[114:115], v[74:75], v[2:3]
	s_nop 0
	v_pk_fma_f32 v[2:3], v[96:97], v[88:89], v[2:3]
	s_nop 0
	v_pk_fma_f32 v[2:3], v[94:95], v[70:71], v[2:3]
	s_nop 0
	v_pk_fma_f32 v[2:3], v[86:87], v[102:103], v[2:3]
	s_nop 0
	v_pk_fma_f32 v[2:3], v[62:63], v[58:59], v[2:3]
	s_nop 0
	v_pk_fma_f32 v[2:3], v[110:111], v[52:53], v[2:3]
	s_nop 0
	v_pk_fma_f32 v[2:3], v[108:109], v[104:105], v[2:3]
	s_nop 0
	v_pk_fma_f32 v[2:3], v[28:29], v[48:49], v[2:3]
	s_nop 0
	v_pk_fma_f32 v[2:3], v[76:77], v[68:69], v[2:3]
	s_nop 0
	v_pk_fma_f32 v[2:3], v[24:25], v[80:81], v[2:3]
	s_nop 0
	v_pk_fma_f32 v[2:3], v[50:51], v[92:93], v[2:3]
	s_nop 0
	v_pk_fma_f32 v[2:3], v[20:21], v[42:43], v[2:3]
	s_nop 0
	v_pk_fma_f32 v[2:3], v[40:41], v[54:55], v[2:3]
	s_nop 0
	v_pk_fma_f32 v[2:3], v[38:39], v[100:101], v[2:3]
	s_nop 0
	v_pk_fma_f32 v[2:3], v[22:23], v[46:47], v[2:3]
	s_nop 0
	v_pk_fma_f32 v[2:3], v[18:19], v[120:121], v[2:3]
	s_nop 0
	v_pk_fma_f32 v[2:3], v[36:37], v[26:27], v[2:3]
	s_waitcnt vmcnt(0)
	v_pk_fma_f32 v[2:3], v[84:85], v[10:11], v[2:3]
	scratch_load_dwordx4 v[8:11], off, off offset:32 ; 16-byte Folded Reload
	scratch_load_dwordx4 v[18:21], off, off offset:48 ; 16-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_pk_fma_f32 v[2:3], v[20:21], v[10:11], v[2:3]
	scratch_load_dwordx4 v[8:11], off, off offset:16 ; 16-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_pk_fma_f32 v[2:3], v[32:33], v[10:11], v[2:3]
	scratch_load_dwordx4 v[8:11], off, off offset:64 ; 16-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_pk_fma_f32 v[2:3], v[10:11], v[118:119], v[2:3]
	s_nop 0
	v_pk_fma_f32 v[2:3], v[112:113], v[98:99], v[2:3]
	s_nop 0
	v_pk_fma_f32 v[2:3], v[90:91], v[82:83], v[2:3]
	s_nop 0
	v_pk_fma_f32 v[2:3], v[78:79], v[30:31], v[2:3]
	s_nop 0
	v_pk_fma_f32 v[2:3], v[14:15], v[16:17], v[2:3]
	s_nop 0
	v_pk_fma_f32 v[2:3], v[12:13], v[6:7], v[2:3]
	s_nop 0
	v_add_f32_e32 v2, v2, v3
	v_fma_f32 v3, v4, v2, -s8
	v_mul_f32_e32 v3, v3, v3
	v_fma_f32 v5, v4, v2, -s9
	v_min_f32_e32 v3, 0x7149f2ca, v3
	v_mul_f32_e32 v5, v5, v5
	v_cmp_lt_f32_e32 vcc, v5, v3
	v_fma_f32 v7, v4, v2, -s11
	v_mul_f32_e32 v7, v7, v7
	v_cndmask_b32_e32 v3, v3, v5, vcc
	v_fma_f32 v5, v4, v2, -s10
	v_mul_f32_e32 v5, v5, v5
	v_min_f32_e32 v6, v5, v3
	v_fma_f32 v9, v4, v2, -s12
	v_fma_f32 v11, v4, v2, -s13
	v_fma_f32 v13, v4, v2, -s14
	v_fma_f32 v15, v4, v2, -s15
	v_fma_f32 v17, v4, v2, -s16
	v_fma_f32 v19, v4, v2, -s17
	v_fma_f32 v21, v4, v2, -s18
	v_fma_f32 v23, v4, v2, -s19
	v_fma_f32 v25, v4, v2, -s20
	v_fma_f32 v27, v4, v2, -s21
	v_fma_f32 v29, v4, v2, -s22
	v_fma_f32 v2, v4, v2, -s23
	v_cndmask_b32_e64 v4, 0, 1, vcc
	v_cmp_ge_f32_e32 vcc, v5, v3
	v_min_f32_e32 v8, v7, v6
	v_mul_f32_e32 v9, v9, v9
	v_cndmask_b32_e32 v3, 2, v4, vcc
	v_cmp_ge_f32_e32 vcc, v7, v6
	v_min_f32_e32 v10, v9, v8
	v_mul_f32_e32 v11, v11, v11
	v_cndmask_b32_e32 v3, 3, v3, vcc
	v_cmp_ge_f32_e32 vcc, v9, v8
	v_min_f32_e32 v12, v11, v10
	v_mul_f32_e32 v13, v13, v13
	v_cndmask_b32_e32 v3, 4, v3, vcc
	v_cmp_ge_f32_e32 vcc, v11, v10
	v_min_f32_e32 v14, v13, v12
	v_mul_f32_e32 v15, v15, v15
	v_cndmask_b32_e32 v3, 5, v3, vcc
	v_cmp_ge_f32_e32 vcc, v13, v12
	v_min_f32_e32 v16, v15, v14
	v_mul_f32_e32 v17, v17, v17
	v_cndmask_b32_e32 v3, 6, v3, vcc
	v_cmp_ge_f32_e32 vcc, v15, v14
	v_min_f32_e32 v18, v17, v16
	v_mul_f32_e32 v19, v19, v19
	v_cndmask_b32_e32 v3, 7, v3, vcc
	v_cmp_ge_f32_e32 vcc, v17, v16
	v_min_f32_e32 v20, v19, v18
	v_mul_f32_e32 v21, v21, v21
	v_cndmask_b32_e32 v3, 8, v3, vcc
	v_cmp_ge_f32_e32 vcc, v19, v18
	v_min_f32_e32 v22, v21, v20
	v_mul_f32_e32 v23, v23, v23
	v_cndmask_b32_e32 v3, 9, v3, vcc
	v_cmp_ge_f32_e32 vcc, v21, v20
	v_min_f32_e32 v24, v23, v22
	v_mul_f32_e32 v25, v25, v25
	v_cndmask_b32_e32 v3, 10, v3, vcc
	v_cmp_ge_f32_e32 vcc, v23, v22
	v_min_f32_e32 v26, v25, v24
	v_mul_f32_e32 v27, v27, v27
	v_cndmask_b32_e32 v3, 11, v3, vcc
	v_cmp_ge_f32_e32 vcc, v25, v24
	v_min_f32_e32 v28, v27, v26
	v_mul_f32_e32 v29, v29, v29
	v_cndmask_b32_e32 v3, 12, v3, vcc
	v_cmp_ge_f32_e32 vcc, v27, v26
	v_min_f32_e32 v30, v29, v28
	v_mul_f32_e32 v2, v2, v2
	v_cndmask_b32_e32 v3, 13, v3, vcc
	v_cmp_ge_f32_e32 vcc, v29, v28
	s_nop 1
	v_cndmask_b32_e32 v3, 14, v3, vcc
	v_cmp_ge_f32_e32 vcc, v2, v30
	s_nop 1
	v_cndmask_b32_e32 v4, 15, v3, vcc
	v_and_b32_e32 v1, 1, v4
	v_lshl_add_u64 v[2:3], s[2:3], 0, v[126:127]
	v_cmp_ne_u32_e64 s[8:9], 0, v1
	s_and_saveexec_b64 s[0:1], s[6:7]
	s_cbranch_execz .LBB3_10
; %bb.9:
	v_mov_b64_e32 v[6:7], s[8:9]
	global_store_dwordx2 v[2:3], v[6:7], off offset:4
.LBB3_10:
	s_or_b64 exec, exec, s[0:1]
	v_bfe_u32 v1, v4, 1, 1
	v_cmp_ne_u32_e64 s[8:9], 0, v1
	s_and_saveexec_b64 s[0:1], s[6:7]
	s_cbranch_execz .LBB3_12
; %bb.11:
	v_mov_b64_e32 v[6:7], s[8:9]
	global_store_dwordx2 v[2:3], v[6:7], off offset:20
.LBB3_12:
	s_or_b64 exec, exec, s[0:1]
	v_bfe_u32 v1, v4, 2, 1
	v_cmp_ne_u32_e64 s[8:9], 0, v1
	s_and_saveexec_b64 s[0:1], s[6:7]
	s_cbranch_execz .LBB3_14
; %bb.13:
	v_mov_b64_e32 v[6:7], s[8:9]
	global_store_dwordx2 v[2:3], v[6:7], off offset:36
.LBB3_14:
	s_or_b64 exec, exec, s[0:1]
	v_bfe_u32 v1, v4, 3, 1
	v_cmp_ne_u32_e64 s[8:9], 0, v1
	s_and_saveexec_b64 s[0:1], s[6:7]
	s_cbranch_execz .LBB3_16
; %bb.15:
	v_mov_b64_e32 v[4:5], s[8:9]
	global_store_dwordx2 v[2:3], v[4:5], off offset:52
.LBB3_16:                               ; %Flow
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[8:9], 0
.LBB3_17:                               ; %Flow83
	s_and_b64 vcc, exec, s[8:9]
	s_cbranch_vccz .LBB3_22
; %bb.18:
	s_and_saveexec_b64 s[0:1], s[4:5]
	s_cbranch_execz .LBB3_20
; %bb.19:
	v_mov_b32_e32 v1, 0
	global_store_dword v1, v1, s[2:3]
.LBB3_20:
	s_or_b64 exec, exec, s[0:1]
	v_cmp_gt_u32_e32 vcc, 8, v0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB3_22
; %bb.21:
	v_lshlrev_b32_e32 v2, 3, v0
	v_mov_b32_e32 v0, 0
	v_mov_b32_e32 v1, v0
	global_store_dwordx2 v2, v[0:1], s[2:3] offset:4
.LBB3_22:                               ; %.loopexit
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z23tqm_quantize_kernel_tq4PKfS0_P16block_tq4_mi300xi
		.amdhsa_group_segment_fixed_size 524
		.amdhsa_private_segment_fixed_size 644
		.amdhsa_kernarg_size 28
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 0
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 128
		.amdhsa_next_free_sgpr 24
		.amdhsa_accum_offset 128
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_tg_split 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end3:
	.size	_Z23tqm_quantize_kernel_tq4PKfS0_P16block_tq4_mi300xi, .Lfunc_end3-_Z23tqm_quantize_kernel_tq4PKfS0_P16block_tq4_mi300xi
                                        ; -- End function
	.set _Z23tqm_quantize_kernel_tq4PKfS0_P16block_tq4_mi300xi.num_vgpr, 128
	.set _Z23tqm_quantize_kernel_tq4PKfS0_P16block_tq4_mi300xi.num_agpr, 0
	.set _Z23tqm_quantize_kernel_tq4PKfS0_P16block_tq4_mi300xi.numbered_sgpr, 24
	.set _Z23tqm_quantize_kernel_tq4PKfS0_P16block_tq4_mi300xi.num_named_barrier, 0
	.set _Z23tqm_quantize_kernel_tq4PKfS0_P16block_tq4_mi300xi.private_seg_size, 644
	.set _Z23tqm_quantize_kernel_tq4PKfS0_P16block_tq4_mi300xi.uses_vcc, 1
	.set _Z23tqm_quantize_kernel_tq4PKfS0_P16block_tq4_mi300xi.uses_flat_scratch, 0
	.set _Z23tqm_quantize_kernel_tq4PKfS0_P16block_tq4_mi300xi.has_dyn_sized_stack, 0
	.set _Z23tqm_quantize_kernel_tq4PKfS0_P16block_tq4_mi300xi.has_recursion, 0
	.set _Z23tqm_quantize_kernel_tq4PKfS0_P16block_tq4_mi300xi.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 3456
; TotalNumSgprs: 30
; NumVgprs: 128
; NumAgprs: 0
; TotalNumVgprs: 128
; ScratchSize: 644
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 524 bytes/workgroup (compile time only)
; SGPRBlocks: 3
; VGPRBlocks: 15
; NumSGPRsForWavesPerEU: 30
; NumVGPRsForWavesPerEU: 128
; AccumOffset: 128
; Occupancy: 4
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 1
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 31
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.protected	_Z25tqm_dequantize_kernel_tq4PK16block_tq4_mi300xPKfPfi ; -- Begin function _Z25tqm_dequantize_kernel_tq4PK16block_tq4_mi300xPKfPfi
	.globl	_Z25tqm_dequantize_kernel_tq4PK16block_tq4_mi300xPKfPfi
	.p2align	8
	.type	_Z25tqm_dequantize_kernel_tq4PK16block_tq4_mi300xPKfPfi,@function
_Z25tqm_dequantize_kernel_tq4PK16block_tq4_mi300xPKfPfi: ; @_Z25tqm_dequantize_kernel_tq4PK16block_tq4_mi300xPKfPfi
; %bb.0:
	s_load_dword s3, s[0:1], 0x18
	s_waitcnt lgkmcnt(0)
	s_cmp_ge_i32 s2, s3
	s_cbranch_scc1 .LBB4_6
; %bb.1:
	s_load_dwordx2 s[6:7], s[0:1], 0x0
	s_load_dwordx2 s[4:5], s[0:1], 0x10
	s_mul_i32 s8, s2, 0x44
	s_mul_hi_i32 s3, s2, 0x44
	v_cmp_eq_u32_e32 vcc, 0, v0
	s_waitcnt lgkmcnt(0)
	s_add_u32 s6, s6, s8
	s_addc_u32 s7, s7, s3
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB4_3
; %bb.2:
	v_mov_b32_e32 v1, 0
	global_load_dword v2, v1, s[6:7]
	s_waitcnt vmcnt(0)
	ds_write_b32 v1, v2 offset:512
.LBB4_3:
	s_or_b64 exec, exec, s[8:9]
	v_mov_b32_e32 v1, 0
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b32 v2, v1 offset:512
	s_mov_b32 s3, 0x26901d7d
	s_waitcnt lgkmcnt(0)
	v_cmp_lt_f32_e64 s[8:9], |v2|, s3
	s_and_b64 vcc, exec, s[8:9]
	s_cbranch_vccnz .LBB4_5
; %bb.4:                                ; %.preheader
	v_lshrrev_b32_e32 v1, 3, v0
	v_and_b32_e32 v1, 0x78, v1
	global_load_dwordx2 v[2:3], v1, s[6:7] offset:4
	global_load_dwordx2 v[4:5], v1, s[6:7] offset:20
	global_load_dwordx2 v[6:7], v1, s[6:7] offset:36
	global_load_dwordx2 v[8:9], v1, s[6:7] offset:52
	v_mov_b32_e32 v11, 0
	s_getpc_b64 s[6:7]
	s_add_u32 s6, s6, d_cb4@rel32@lo+4
	s_addc_u32 s7, s7, d_cb4@rel32@hi+12
	s_load_dwordx2 s[0:1], s[0:1], 0x8
	s_waitcnt vmcnt(3)
	v_lshrrev_b64 v[2:3], v0, v[2:3]
	s_waitcnt vmcnt(2)
	v_lshrrev_b64 v[4:5], v0, v[4:5]
	v_and_b32_e32 v1, 1, v2
	s_waitcnt vmcnt(1)
	v_lshrrev_b64 v[6:7], v0, v[6:7]
	v_lshlrev_b32_e32 v4, 3, v4
	v_lshlrev_b32_e32 v10, 2, v1
	s_waitcnt vmcnt(0)
	v_lshrrev_b64 v[8:9], v0, v[8:9]
	v_lshlrev_b32_e32 v5, 4, v6
	v_lshl_add_u64 v[2:3], s[6:7], 0, v[10:11]
	v_and_b32_e32 v10, 8, v4
	v_lshlrev_b32_e32 v6, 5, v8
	v_lshl_add_u64 v[2:3], v[2:3], 0, v[10:11]
	v_and_b32_e32 v10, 16, v5
	v_lshl_add_u64 v[2:3], v[2:3], 0, v[10:11]
	v_and_b32_e32 v10, 32, v6
	v_lshl_add_u64 v[2:3], v[2:3], 0, v[10:11]
	global_load_dword v2, v[2:3], off
	v_lshlrev_b32_e32 v1, 2, v0
	v_add_u32_e32 v3, 0x200, v1
	v_add_u32_e32 v4, 0x400, v1
	v_add_u32_e32 v5, 0x600, v1
	v_or_b32_e32 v6, 0x1000, v1
	v_add_u32_e32 v7, 0x3200, v1
	v_add_u32_e32 v10, 0x3400, v1
	v_add_u32_e32 v40, 0x3800, v1
	v_add_u32_e32 v41, 0x3a00, v1
	v_add_u32_e32 v42, 0x3e00, v1
	v_or_b32_e32 v43, 0x4000, v1
	v_add_u32_e32 v44, 0x4200, v1
	v_add_u32_e32 v45, 0x4400, v1
	v_add_u32_e32 v46, 0x4600, v1
	v_add_u32_e32 v47, 0x4800, v1
	v_add_u32_e32 v48, 0x4a00, v1
	v_add_u32_e32 v49, 0x4c00, v1
	v_add_u32_e32 v50, 0x4e00, v1
	v_or_b32_e32 v51, 0x5000, v1
	v_add_u32_e32 v52, 0x5200, v1
	v_add_u32_e32 v53, 0x5400, v1
	v_add_u32_e32 v54, 0x5600, v1
	v_add_u32_e32 v55, 0x5800, v1
	v_add_u32_e32 v56, 0x5a00, v1
	v_add_u32_e32 v57, 0x5c00, v1
	v_add_u32_e32 v58, 0x5e00, v1
	v_or_b32_e32 v59, 0x6000, v1
	v_add_u32_e32 v60, 0x6200, v1
	v_add_u32_e32 v61, 0x6400, v1
	v_add_u32_e32 v62, 0x6600, v1
	v_add_u32_e32 v63, 0x6800, v1
	v_add_u32_e32 v64, 0x7600, v1
	v_or_b32_e32 v65, 0x8000, v1
	v_add_u32_e32 v66, 0x8200, v1
	v_add_u32_e32 v67, 0x8400, v1
	v_add_u32_e32 v68, 0x8600, v1
	v_add_u32_e32 v69, 0x8800, v1
	v_add_u32_e32 v70, 0x8a00, v1
	v_add_u32_e32 v71, 0x8e00, v1
	v_or_b32_e32 v72, 0x9000, v1
	v_add_u32_e32 v73, 0x9200, v1
	v_add_u32_e32 v74, 0x9400, v1
	s_waitcnt vmcnt(0)
	ds_write_b32 v1, v2
	s_waitcnt lgkmcnt(0)
	s_barrier
	global_load_dword v26, v1, s[0:1]
	global_load_dword v27, v3, s[0:1]
	global_load_dword v28, v4, s[0:1]
	global_load_dword v29, v5, s[0:1]
	v_add_u32_e32 v2, 0x800, v1
	v_add_u32_e32 v3, 0xa00, v1
	v_add_u32_e32 v4, 0xc00, v1
	v_add_u32_e32 v5, 0xe00, v1
	global_load_dword v14, v2, s[0:1]
	global_load_dword v15, v3, s[0:1]
	global_load_dword v16, v4, s[0:1]
	global_load_dword v17, v5, s[0:1]
	v_add_u32_e32 v2, 0x1200, v1
	v_add_u32_e32 v3, 0x1400, v1
	global_load_dword v30, v6, s[0:1]
	global_load_dword v31, v2, s[0:1]
	v_add_u32_e32 v4, 0x1600, v1
	global_load_dword v22, v3, s[0:1]
	global_load_dword v23, v4, s[0:1]
	v_add_u32_e32 v5, 0x1800, v1
	v_add_u32_e32 v2, 0x1a00, v1
	v_add_u32_e32 v3, 0x1c00, v1
	v_add_u32_e32 v4, 0x1e00, v1
	global_load_dword v20, v5, s[0:1]
	global_load_dword v21, v2, s[0:1]
	global_load_dword v18, v3, s[0:1]
	global_load_dword v19, v4, s[0:1]
	v_or_b32_e32 v6, 0x2000, v1
	v_add_u32_e32 v2, 0x2200, v1
	v_add_u32_e32 v3, 0x2400, v1
	global_load_dword v24, v6, s[0:1]
	global_load_dword v25, v2, s[0:1]
	v_add_u32_e32 v2, 0x2600, v1
	global_load_dword v12, v3, s[0:1]
	global_load_dword v13, v2, s[0:1]
	v_add_u32_e32 v2, 0x2800, v1
	v_add_u32_e32 v3, 0x2a00, v1
	v_add_u32_e32 v4, 0x2c00, v1
	v_add_u32_e32 v5, 0x2e00, v1
	global_load_dword v2, v2, s[0:1]
	global_load_dword v3, v3, s[0:1]
	global_load_dword v4, v4, s[0:1]
	global_load_dword v5, v5, s[0:1]
	v_or_b32_e32 v6, 0x3000, v1
	global_load_dword v8, v6, s[0:1]
	global_load_dword v9, v7, s[0:1]
	v_add_u32_e32 v7, 0x3600, v1
	global_load_dword v6, v10, s[0:1]
	global_load_dword v7, v7, s[0:1]
	ds_read_b128 v[32:35], v11
	ds_read_b128 v[36:39], v11 offset:16
	v_add_u32_e32 v10, 0x3c00, v1
	s_waitcnt vmcnt(26) lgkmcnt(1)
	v_pk_mul_f32 v[26:27], v[32:33], v[26:27]
	s_waitcnt vmcnt(24)
	v_pk_fma_f32 v[34:35], v[34:35], v[28:29], v[26:27]
	global_load_dword v32, v40, s[0:1]
	global_load_dword v33, v41, s[0:1]
	global_load_dword v28, v10, s[0:1]
	global_load_dword v29, v42, s[0:1]
	global_load_dword v26, v43, s[0:1]
	global_load_dword v27, v44, s[0:1]
	v_add_u32_e32 v10, 0x6a00, v1
	s_waitcnt vmcnt(28) lgkmcnt(0)
	v_pk_fma_f32 v[14:15], v[36:37], v[14:15], v[34:35]
	ds_read_b128 v[34:37], v11 offset:32
	s_waitcnt vmcnt(26)
	v_pk_fma_f32 v[16:17], v[38:39], v[16:17], v[14:15]
	global_load_dword v14, v45, s[0:1]
	global_load_dword v15, v46, s[0:1]
	ds_read_b128 v[38:41], v11 offset:48
	v_add_u32_e32 v42, 0x6c00, v1
	s_waitcnt vmcnt(26) lgkmcnt(1)
	v_pk_fma_f32 v[16:17], v[34:35], v[30:31], v[16:17]
	v_add_u32_e32 v43, 0x6e00, v1
	s_waitcnt vmcnt(24)
	v_pk_fma_f32 v[16:17], v[36:37], v[22:23], v[16:17]
	v_or_b32_e32 v44, 0x7000, v1
	s_waitcnt vmcnt(22) lgkmcnt(0)
	v_pk_fma_f32 v[16:17], v[38:39], v[20:21], v[16:17]
	ds_read_b128 v[20:23], v11 offset:64
	s_waitcnt vmcnt(20)
	v_pk_fma_f32 v[30:31], v[40:41], v[18:19], v[16:17]
	ds_read_b128 v[16:19], v11 offset:80
	v_add_u32_e32 v45, 0x7200, v1
	v_add_u32_e32 v46, 0x7400, v1
	s_waitcnt vmcnt(18) lgkmcnt(1)
	v_pk_fma_f32 v[20:21], v[20:21], v[24:25], v[30:31]
	v_add_u32_e32 v38, 0x7800, v1
	s_waitcnt vmcnt(16)
	v_pk_fma_f32 v[12:13], v[22:23], v[12:13], v[20:21]
	v_add_u32_e32 v39, 0x7a00, v1
	v_add_u32_e32 v40, 0x7c00, v1
	s_waitcnt vmcnt(14) lgkmcnt(0)
	v_pk_fma_f32 v[2:3], v[16:17], v[2:3], v[12:13]
	v_add_u32_e32 v41, 0x7e00, v1
	s_waitcnt vmcnt(12)
	v_pk_fma_f32 v[12:13], v[18:19], v[4:5], v[2:3]
	ds_read_b128 v[2:5], v11 offset:96
	global_load_dword v24, v47, s[0:1]
	global_load_dword v25, v48, s[0:1]
	global_load_dword v48, v49, s[0:1]
	global_load_dword v49, v50, s[0:1]
	global_load_dword v50, v51, s[0:1]
	global_load_dword v51, v52, s[0:1]
	ds_read_b128 v[34:37], v11 offset:112
	global_load_dword v52, v53, s[0:1]
	global_load_dword v53, v54, s[0:1]
	global_load_dword v54, v55, s[0:1]
	global_load_dword v55, v56, s[0:1]
	global_load_dword v56, v57, s[0:1]
	global_load_dword v57, v58, s[0:1]
	global_load_dword v58, v59, s[0:1]
	global_load_dword v59, v60, s[0:1]
	s_waitcnt vmcnt(24) lgkmcnt(1)
	v_pk_fma_f32 v[2:3], v[2:3], v[8:9], v[12:13]
	v_add_u32_e32 v47, 0x8c00, v1
	s_waitcnt vmcnt(22)
	v_pk_fma_f32 v[30:31], v[4:5], v[6:7], v[2:3]
	global_load_dword v22, v61, s[0:1]
	global_load_dword v23, v62, s[0:1]
	global_load_dword v20, v63, s[0:1]
	global_load_dword v21, v10, s[0:1]
	global_load_dword v18, v42, s[0:1]
	global_load_dword v19, v43, s[0:1]
	global_load_dword v6, v44, s[0:1]
	global_load_dword v7, v45, s[0:1]
	global_load_dword v16, v46, s[0:1]
	global_load_dword v17, v64, s[0:1]
	global_load_dword v4, v38, s[0:1]
	global_load_dword v5, v39, s[0:1]
	global_load_dword v12, v40, s[0:1]
	global_load_dword v13, v41, s[0:1]
	global_load_dword v2, v65, s[0:1]
	global_load_dword v3, v66, s[0:1]
	global_load_dword v8, v67, s[0:1]
	global_load_dword v9, v68, s[0:1]
	v_or_b32_e32 v38, 0xa000, v1
	v_add_u32_e32 v39, 0xa200, v1
	v_add_u32_e32 v60, 0x9600, v1
	v_add_u32_e32 v10, 0x9800, v1
	v_add_u32_e32 v61, 0xac00, v1
	v_add_u32_e32 v62, 0xae00, v1
	v_or_b32_e32 v63, 0xb000, v1
	v_add_u32_e32 v64, 0xb200, v1
	v_add_u32_e32 v65, 0xb400, v1
	v_add_u32_e32 v66, 0xb600, v1
	v_add_u32_e32 v67, 0xc400, v1
	v_add_u32_e32 v68, 0xc600, v1
	ds_read_b128 v[40:43], v11 offset:144
	s_waitcnt vmcnt(38) lgkmcnt(1)
	v_pk_fma_f32 v[34:35], v[34:35], v[32:33], v[30:31]
	ds_read_b128 v[30:33], v11 offset:128
	s_waitcnt vmcnt(36)
	v_pk_fma_f32 v[28:29], v[36:37], v[28:29], v[34:35]
	v_add_u32_e32 v35, 0x9a00, v1
	v_add_u32_e32 v36, 0x9c00, v1
	v_add_u32_e32 v37, 0x9e00, v1
	s_waitcnt vmcnt(34) lgkmcnt(0)
	v_pk_fma_f32 v[26:27], v[30:31], v[26:27], v[28:29]
	s_waitcnt vmcnt(32)
	v_pk_fma_f32 v[14:15], v[32:33], v[14:15], v[26:27]
	global_load_dword v26, v69, s[0:1]
	global_load_dword v27, v70, s[0:1]
	global_load_dword v28, v47, s[0:1]
	global_load_dword v29, v71, s[0:1]
	global_load_dword v30, v72, s[0:1]
	global_load_dword v31, v73, s[0:1]
	global_load_dword v32, v74, s[0:1]
	global_load_dword v33, v60, s[0:1]
	global_load_dword v34, v10, s[0:1]
	global_load_dword v35, v35, s[0:1]
	global_load_dword v36, v36, s[0:1]
	global_load_dword v37, v37, s[0:1]
	global_load_dword v38, v38, s[0:1]
	global_load_dword v39, v39, s[0:1]
	ds_read_b128 v[44:47], v11 offset:160
	v_add_u32_e32 v10, 0xa400, v1
	v_add_u32_e32 v60, 0xa600, v1
	v_add_u32_e32 v69, 0xc800, v1
	s_waitcnt vmcnt(44)
	v_pk_fma_f32 v[14:15], v[40:41], v[24:25], v[14:15]
	v_add_u32_e32 v24, 0xa800, v1
	s_waitcnt vmcnt(42)
	v_pk_fma_f32 v[14:15], v[42:43], v[48:49], v[14:15]
	ds_read_b128 v[40:43], v11 offset:176
	s_waitcnt vmcnt(40) lgkmcnt(1)
	v_pk_fma_f32 v[14:15], v[44:45], v[50:51], v[14:15]
	v_add_u32_e32 v25, 0xaa00, v1
	s_waitcnt vmcnt(38)
	v_pk_fma_f32 v[14:15], v[46:47], v[52:53], v[14:15]
	ds_read_b128 v[44:47], v11 offset:192
	s_waitcnt vmcnt(36) lgkmcnt(1)
	v_pk_fma_f32 v[14:15], v[40:41], v[54:55], v[14:15]
	v_add_u32_e32 v54, 0xb800, v1
	s_waitcnt vmcnt(34)
	v_pk_fma_f32 v[14:15], v[42:43], v[56:57], v[14:15]
	ds_read_b128 v[40:43], v11 offset:208
	s_waitcnt vmcnt(32) lgkmcnt(1)
	v_pk_fma_f32 v[14:15], v[44:45], v[58:59], v[14:15]
	v_add_u32_e32 v55, 0xba00, v1
	s_waitcnt vmcnt(30)
	v_pk_fma_f32 v[14:15], v[46:47], v[22:23], v[14:15]
	v_add_u32_e32 v56, 0xbc00, v1
	s_waitcnt vmcnt(28) lgkmcnt(0)
	v_pk_fma_f32 v[14:15], v[40:41], v[20:21], v[14:15]
	ds_read_b128 v[20:23], v11 offset:224
	s_waitcnt vmcnt(26)
	v_pk_fma_f32 v[14:15], v[42:43], v[18:19], v[14:15]
	ds_read_b128 v[40:43], v11 offset:240
	v_add_u32_e32 v57, 0xbe00, v1
	v_or_b32_e32 v58, 0xc000, v1
	s_waitcnt vmcnt(24) lgkmcnt(1)
	v_pk_fma_f32 v[6:7], v[20:21], v[6:7], v[14:15]
	v_add_u32_e32 v59, 0xc200, v1
	s_waitcnt vmcnt(22)
	v_pk_fma_f32 v[6:7], v[22:23], v[16:17], v[6:7]
	ds_read_b128 v[14:17], v11 offset:256
	ds_read_b128 v[44:47], v11 offset:272
	s_waitcnt vmcnt(20) lgkmcnt(2)
	v_pk_fma_f32 v[4:5], v[40:41], v[4:5], v[6:7]
	s_waitcnt vmcnt(18)
	v_pk_fma_f32 v[4:5], v[42:43], v[12:13], v[4:5]
	ds_read_b128 v[40:43], v11 offset:288
	ds_read_b128 v[48:51], v11 offset:304
	s_waitcnt vmcnt(16) lgkmcnt(3)
	v_pk_fma_f32 v[2:3], v[14:15], v[2:3], v[4:5]
	s_waitcnt vmcnt(14)
	v_pk_fma_f32 v[52:53], v[16:17], v[8:9], v[2:3]
	ds_read_b128 v[6:9], v11 offset:320
	ds_read_b128 v[2:5], v11 offset:336
	global_load_dword v16, v10, s[0:1]
	global_load_dword v17, v60, s[0:1]
	global_load_dword v14, v24, s[0:1]
	global_load_dword v15, v25, s[0:1]
	global_load_dword v12, v61, s[0:1]
	global_load_dword v13, v62, s[0:1]
	global_load_dword v20, v63, s[0:1]
	global_load_dword v21, v64, s[0:1]
	global_load_dword v18, v65, s[0:1]
	global_load_dword v19, v66, s[0:1]
	global_load_dword v24, v54, s[0:1]
	global_load_dword v25, v55, s[0:1]
	global_load_dword v22, v56, s[0:1]
	global_load_dword v23, v57, s[0:1]
	v_add_u32_e32 v10, 0xca00, v1
	v_add_u32_e32 v54, 0xf800, v1
	v_add_u32_e32 v55, 0xfa00, v1
	v_add_u32_e32 v56, 0xfc00, v1
	s_waitcnt vmcnt(26) lgkmcnt(4)
	v_pk_fma_f32 v[26:27], v[44:45], v[26:27], v[52:53]
	v_add_u32_e32 v44, 0xe400, v1
	s_waitcnt vmcnt(24)
	v_pk_fma_f32 v[26:27], v[46:47], v[28:29], v[26:27]
	v_add_u32_e32 v45, 0xe600, v1
	s_waitcnt vmcnt(22) lgkmcnt(3)
	v_pk_fma_f32 v[26:27], v[40:41], v[30:31], v[26:27]
	v_add_u32_e32 v30, 0xcc00, v1
	s_waitcnt vmcnt(20)
	v_pk_fma_f32 v[26:27], v[42:43], v[32:33], v[26:27]
	v_add_u32_e32 v31, 0xce00, v1
	s_waitcnt vmcnt(18) lgkmcnt(2)
	v_pk_fma_f32 v[26:27], v[48:49], v[34:35], v[26:27]
	v_add_u32_e32 v35, 0xd200, v1
	s_waitcnt vmcnt(16)
	v_pk_fma_f32 v[26:27], v[50:51], v[36:37], v[26:27]
	v_add_u32_e32 v36, 0xd400, v1
	s_waitcnt vmcnt(14) lgkmcnt(1)
	v_pk_fma_f32 v[6:7], v[6:7], v[38:39], v[26:27]
	global_load_dword v28, v58, s[0:1]
	global_load_dword v29, v59, s[0:1]
	global_load_dword v26, v67, s[0:1]
	global_load_dword v27, v68, s[0:1]
	global_load_dword v32, v69, s[0:1]
	global_load_dword v33, v10, s[0:1]
	global_load_dword v30, v30, s[0:1]
	global_load_dword v31, v31, s[0:1]
	v_or_b32_e32 v10, 0xd000, v1
	v_add_u32_e32 v37, 0xd600, v1
	v_add_u32_e32 v38, 0xd800, v1
	v_add_u32_e32 v39, 0xda00, v1
	v_add_u32_e32 v40, 0xdc00, v1
	v_add_u32_e32 v41, 0xde00, v1
	global_load_dword v34, v10, s[0:1]
	global_load_dword v35, v35, s[0:1]
	global_load_dword v36, v36, s[0:1]
	global_load_dword v37, v37, s[0:1]
	global_load_dword v38, v38, s[0:1]
	global_load_dword v39, v39, s[0:1]
	global_load_dword v40, v40, s[0:1]
	global_load_dword v41, v41, s[0:1]
	v_or_b32_e32 v10, 0xe000, v1
	v_add_u32_e32 v43, 0xe200, v1
	v_add_u32_e32 v46, 0xe800, v1
	v_add_u32_e32 v47, 0xea00, v1
	v_add_u32_e32 v48, 0xec00, v1
	v_add_u32_e32 v49, 0xee00, v1
	global_load_dword v42, v10, s[0:1]
	global_load_dword v43, v43, s[0:1]
	global_load_dword v44, v44, s[0:1]
	global_load_dword v45, v45, s[0:1]
	global_load_dword v46, v46, s[0:1]
	global_load_dword v47, v47, s[0:1]
	global_load_dword v48, v48, s[0:1]
	global_load_dword v49, v49, s[0:1]
	v_or_b32_e32 v10, 0xf000, v1
	v_add_u32_e32 v51, 0xf200, v1
	v_add_u32_e32 v52, 0xf400, v1
	v_add_u32_e32 v53, 0xf600, v1
	v_add_u32_e32 v1, 0xfe00, v1
	global_load_dword v50, v10, s[0:1]
	global_load_dword v51, v51, s[0:1]
	global_load_dword v52, v52, s[0:1]
	global_load_dword v53, v53, s[0:1]
	global_load_dword v54, v54, s[0:1]
	global_load_dword v55, v55, s[0:1]
	global_load_dword v56, v56, s[0:1]
	global_load_dword v57, v1, s[0:1]
	ds_read_b32 v1, v11 offset:512
	s_waitcnt vmcnt(44)
	v_pk_fma_f32 v[16:17], v[8:9], v[16:17], v[6:7]
	ds_read_b128 v[6:9], v11 offset:352
	s_waitcnt vmcnt(42) lgkmcnt(2)
	v_pk_fma_f32 v[2:3], v[2:3], v[14:15], v[16:17]
	ds_read_b128 v[14:17], v11 offset:368
	s_waitcnt vmcnt(40)
	v_pk_fma_f32 v[2:3], v[4:5], v[12:13], v[2:3]
	s_waitcnt vmcnt(38) lgkmcnt(1)
	v_pk_fma_f32 v[6:7], v[6:7], v[20:21], v[2:3]
	ds_read_b128 v[2:5], v11 offset:384
	s_waitcnt vmcnt(36)
	v_pk_fma_f32 v[6:7], v[8:9], v[18:19], v[6:7]
	s_waitcnt vmcnt(34) lgkmcnt(1)
	v_pk_fma_f32 v[12:13], v[14:15], v[24:25], v[6:7]
	ds_read_b128 v[6:9], v11 offset:400
	s_waitcnt vmcnt(32)
	v_pk_fma_f32 v[12:13], v[16:17], v[22:23], v[12:13]
	s_waitcnt vmcnt(30) lgkmcnt(1)
	v_pk_fma_f32 v[2:3], v[2:3], v[28:29], v[12:13]
	ds_read_b128 v[12:15], v11 offset:416
	s_waitcnt vmcnt(28)
	v_pk_fma_f32 v[2:3], v[4:5], v[26:27], v[2:3]
	s_waitcnt vmcnt(26) lgkmcnt(1)
	v_pk_fma_f32 v[6:7], v[6:7], v[32:33], v[2:3]
	ds_read_b128 v[2:5], v11 offset:432
	s_waitcnt vmcnt(24)
	v_pk_fma_f32 v[6:7], v[8:9], v[30:31], v[6:7]
	s_waitcnt vmcnt(22) lgkmcnt(1)
	v_pk_fma_f32 v[12:13], v[12:13], v[34:35], v[6:7]
	ds_read_b128 v[6:9], v11 offset:448
	s_waitcnt vmcnt(20)
	v_pk_fma_f32 v[12:13], v[14:15], v[36:37], v[12:13]
	s_waitcnt vmcnt(18) lgkmcnt(1)
	v_pk_fma_f32 v[2:3], v[2:3], v[38:39], v[12:13]
	ds_read_b128 v[12:15], v11 offset:464
	s_waitcnt vmcnt(16)
	v_pk_fma_f32 v[2:3], v[4:5], v[40:41], v[2:3]
	s_waitcnt vmcnt(14) lgkmcnt(1)
	v_pk_fma_f32 v[6:7], v[6:7], v[42:43], v[2:3]
	ds_read_b128 v[2:5], v11 offset:480
	s_waitcnt vmcnt(12)
	v_pk_fma_f32 v[6:7], v[8:9], v[44:45], v[6:7]
	s_waitcnt vmcnt(10) lgkmcnt(1)
	v_pk_fma_f32 v[12:13], v[12:13], v[46:47], v[6:7]
	ds_read_b128 v[6:9], v11 offset:496
	s_waitcnt vmcnt(8)
	v_pk_fma_f32 v[12:13], v[14:15], v[48:49], v[12:13]
	s_waitcnt vmcnt(6) lgkmcnt(1)
	v_pk_fma_f32 v[2:3], v[2:3], v[50:51], v[12:13]
	s_waitcnt vmcnt(4)
	v_pk_fma_f32 v[2:3], v[4:5], v[52:53], v[2:3]
	s_waitcnt vmcnt(2) lgkmcnt(0)
	v_pk_fma_f32 v[2:3], v[6:7], v[54:55], v[2:3]
	s_waitcnt vmcnt(0)
	v_pk_fma_f32 v[2:3], v[8:9], v[56:57], v[2:3]
	s_nop 0
	v_add_f32_e32 v2, v2, v3
	v_mul_f32_e32 v1, v1, v2
.LBB4_5:                                ; %.sink.split
	v_lshl_add_u32 v2, s2, 7, v0
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[2:3], v[2:3], 2, s[4:5]
	global_store_dword v[2:3], v1, off
.LBB4_6:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z25tqm_dequantize_kernel_tq4PK16block_tq4_mi300xPKfPfi
		.amdhsa_group_segment_fixed_size 516
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 28
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 0
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 75
		.amdhsa_next_free_sgpr 10
		.amdhsa_accum_offset 76
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_tg_split 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end4:
	.size	_Z25tqm_dequantize_kernel_tq4PK16block_tq4_mi300xPKfPfi, .Lfunc_end4-_Z25tqm_dequantize_kernel_tq4PK16block_tq4_mi300xPKfPfi
                                        ; -- End function
	.set _Z25tqm_dequantize_kernel_tq4PK16block_tq4_mi300xPKfPfi.num_vgpr, 75
	.set _Z25tqm_dequantize_kernel_tq4PK16block_tq4_mi300xPKfPfi.num_agpr, 0
	.set _Z25tqm_dequantize_kernel_tq4PK16block_tq4_mi300xPKfPfi.numbered_sgpr, 10
	.set _Z25tqm_dequantize_kernel_tq4PK16block_tq4_mi300xPKfPfi.num_named_barrier, 0
	.set _Z25tqm_dequantize_kernel_tq4PK16block_tq4_mi300xPKfPfi.private_seg_size, 0
	.set _Z25tqm_dequantize_kernel_tq4PK16block_tq4_mi300xPKfPfi.uses_vcc, 1
	.set _Z25tqm_dequantize_kernel_tq4PK16block_tq4_mi300xPKfPfi.uses_flat_scratch, 0
	.set _Z25tqm_dequantize_kernel_tq4PK16block_tq4_mi300xPKfPfi.has_dyn_sized_stack, 0
	.set _Z25tqm_dequantize_kernel_tq4PK16block_tq4_mi300xPKfPfi.has_recursion, 0
	.set _Z25tqm_dequantize_kernel_tq4PK16block_tq4_mi300xPKfPfi.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 3488
; TotalNumSgprs: 16
; NumVgprs: 75
; NumAgprs: 0
; TotalNumVgprs: 75
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 516 bytes/workgroup (compile time only)
; SGPRBlocks: 1
; VGPRBlocks: 9
; NumSGPRsForWavesPerEU: 16
; NumVGPRsForWavesPerEU: 75
; AccumOffset: 76
; Occupancy: 6
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 18
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.protected	_Z14tqm_qjl_kernelPKfPK16block_tq3_mi300xS0_S0_P16block_qjl_mi300xi ; -- Begin function _Z14tqm_qjl_kernelPKfPK16block_tq3_mi300xS0_S0_P16block_qjl_mi300xi
	.globl	_Z14tqm_qjl_kernelPKfPK16block_tq3_mi300xS0_S0_P16block_qjl_mi300xi
	.p2align	8
	.type	_Z14tqm_qjl_kernelPKfPK16block_tq3_mi300xS0_S0_P16block_qjl_mi300xi,@function
_Z14tqm_qjl_kernelPKfPK16block_tq3_mi300xS0_S0_P16block_qjl_mi300xi: ; @_Z14tqm_qjl_kernelPKfPK16block_tq3_mi300xS0_S0_P16block_qjl_mi300xi
; %bb.0:
	s_load_dword s3, s[0:1], 0x28
	s_waitcnt lgkmcnt(0)
	s_cmp_ge_i32 s2, s3
	s_cbranch_scc1 .LBB5_9
; %bb.1:                                ; %vector.ph
	s_load_dwordx8 s[4:11], s[0:1], 0x0
	s_mul_i32 s12, s2, 52
	v_lshrrev_b32_e32 v1, 6, v0
	s_mul_hi_i32 s3, s2, 52
	v_lshlrev_b32_e32 v1, 3, v1
	s_waitcnt lgkmcnt(0)
	s_add_u32 s6, s6, s12
	s_addc_u32 s7, s7, s3
	global_load_dwordx2 v[4:5], v1, s[6:7] offset:4
	global_load_dwordx2 v[6:7], v1, s[6:7] offset:20
	global_load_dwordx2 v[8:9], v1, s[6:7] offset:36
	v_mov_b32_e32 v3, 0
	s_getpc_b64 s[6:7]
	s_add_u32 s6, s6, d_cb3@rel32@lo+4
	s_addc_u32 s7, s7, d_cb3@rel32@hi+12
	scratch_store_dword off, v1, off offset:512 ; 4-byte Folded Spill
	s_waitcnt vmcnt(3)
	v_lshrrev_b64 v[4:5], v0, v[4:5]
	s_waitcnt vmcnt(2)
	v_lshrrev_b64 v[6:7], v0, v[6:7]
	v_and_b32_e32 v2, 1, v4
	s_waitcnt vmcnt(1)
	v_lshrrev_b64 v[8:9], v0, v[8:9]
	v_lshlrev_b32_e32 v6, 3, v6
	v_lshlrev_b32_e32 v2, 2, v2
	v_lshlrev_b32_e32 v7, 4, v8
	v_lshl_add_u64 v[4:5], s[6:7], 0, v[2:3]
	v_and_b32_e32 v2, 8, v6
	v_lshl_add_u64 v[4:5], v[4:5], 0, v[2:3]
	v_and_b32_e32 v2, 16, v7
	v_lshl_add_u64 v[4:5], v[4:5], 0, v[2:3]
	global_load_dword v4, v[4:5], off
	v_lshlrev_b32_e32 v2, 2, v0
	v_add_u32_e32 v5, 0x200, v2
	v_add_u32_e32 v6, 0x400, v2
	v_add_u32_e32 v7, 0x600, v2
	v_add_u32_e32 v8, 0x800, v2
	v_add_u32_e32 v9, 0xa00, v2
	v_add_u32_e32 v10, 0xc00, v2
	v_add_u32_e32 v11, 0xe00, v2
	v_or_b32_e32 v14, 0x1000, v2
	v_add_u32_e32 v15, 0x1200, v2
	v_add_u32_e32 v66, 0x1400, v2
	v_add_u32_e32 v67, 0x1600, v2
	v_add_u32_e32 v104, 0x1800, v2
	v_add_u32_e32 v105, 0x1a00, v2
	v_add_u32_e32 v106, 0x1c00, v2
	v_add_u32_e32 v107, 0x1e00, v2
	v_add_u32_e32 v114, 0x2c00, v2
	v_add_u32_e32 v115, 0x2e00, v2
	v_or_b32_e32 v16, 0x3000, v2
	v_add_u32_e32 v17, 0x3200, v2
	v_add_u32_e32 v18, 0x3400, v2
	v_add_u32_e32 v19, 0x3600, v2
	v_or_b32_e32 v20, 0x2000, v2
	v_add_u32_e32 v21, 0x2200, v2
	v_add_u32_e32 v22, 0x2400, v2
	v_add_u32_e32 v23, 0x2600, v2
	v_add_u32_e32 v96, 0x2800, v2
	v_add_u32_e32 v97, 0x2a00, v2
	v_add_u32_e32 v98, 0x3800, v2
	v_add_u32_e32 v99, 0x3a00, v2
	v_add_u32_e32 v116, 0x3c00, v2
	v_add_u32_e32 v117, 0x3e00, v2
	v_add_u32_e32 v31, 0x4200, v2
	v_add_u32_e32 v38, 0x4400, v2
	v_add_u32_e32 v39, 0x4600, v2
	v_add_u32_e32 v30, 0x4800, v2
	v_add_u32_e32 v29, 0x4a00, v2
	v_add_u32_e32 v27, 0x4c00, v2
	v_add_u32_e32 v28, 0x4e00, v2
	v_or_b32_e32 v26, 0x5000, v2
	v_add_u32_e32 v25, 0x5200, v2
	v_add_u32_e32 v24, 0x5400, v2
	v_add_u32_e32 v47, 0x5600, v2
	v_add_u32_e32 v48, 0x5800, v2
	v_add_u32_e32 v49, 0x5a00, v2
	v_add_u32_e32 v50, 0x5c00, v2
	v_add_u32_e32 v51, 0x5e00, v2
	v_or_b32_e32 v52, 0x6000, v2
	v_add_u32_e32 v53, 0x6200, v2
	v_add_u32_e32 v54, 0x6400, v2
	v_add_u32_e32 v55, 0x6600, v2
	v_add_u32_e32 v34, 0x6800, v2
	v_add_u32_e32 v35, 0x6a00, v2
	v_add_u32_e32 v32, 0x6c00, v2
	v_add_u32_e32 v33, 0x6e00, v2
	v_or_b32_e32 v36, 0x7000, v2
	v_add_u32_e32 v37, 0x7200, v2
	v_add_u32_e32 v62, 0x7400, v2
	v_add_u32_e32 v63, 0x7600, v2
	v_add_u32_e32 v40, 0x7800, v2
	v_add_u32_e32 v41, 0x7a00, v2
	v_add_u32_e32 v46, 0x7c00, v2
	v_add_u32_e32 v45, 0x7e00, v2
	v_or_b32_e32 v68, 0x8000, v2
	v_add_u32_e32 v69, 0x8200, v2
	v_add_u32_e32 v70, 0x8400, v2
	v_add_u32_e32 v71, 0x8600, v2
	v_add_u32_e32 v44, 0x8800, v2
	v_add_u32_e32 v73, 0x8a00, v2
	v_add_u32_e32 v74, 0x8c00, v2
	v_add_u32_e32 v75, 0x8e00, v2
	v_or_b32_e32 v42, 0x9000, v2
	v_add_u32_e32 v43, 0x9200, v2
	v_add_u32_e32 v78, 0x9400, v2
	v_add_u32_e32 v79, 0x9600, v2
	v_add_u32_e32 v56, 0x9800, v2
	v_add_u32_e32 v57, 0x9a00, v2
	v_add_u32_e32 v61, 0x9c00, v2
	v_add_u32_e32 v60, 0x9e00, v2
	s_waitcnt vmcnt(0)
	ds_write_b32 v2, v4
	s_waitcnt lgkmcnt(0)
	s_barrier
	global_load_dword v4, v2, s[8:9]
	global_load_dword v5, v5, s[8:9]
	global_load_dword v6, v6, s[8:9]
	global_load_dword v7, v7, s[8:9]
	global_load_dword v8, v8, s[8:9]
	global_load_dword v9, v9, s[8:9]
	global_load_dword v12, v10, s[8:9]
	global_load_dword v13, v11, s[8:9]
	global_load_dword v10, v14, s[8:9]
	global_load_dword v11, v15, s[8:9]
	global_load_dword v14, v66, s[8:9]
	global_load_dword v15, v67, s[8:9]
	ds_read_b128 v[88:91], v3
	ds_read_b128 v[92:95], v3 offset:16
	v_or_b32_e32 v84, 0xa000, v2
	v_add_u32_e32 v85, 0xa200, v2
	v_add_u32_e32 v59, 0xa400, v2
	v_add_u32_e32 v58, 0xa600, v2
	v_or_b32_e32 v66, 0xb000, v2
	v_add_u32_e32 v67, 0xb200, v2
	v_add_u32_e32 v64, 0xb400, v2
	v_add_u32_e32 v65, 0xb600, v2
	v_add_u32_e32 v72, 0xb800, v2
	v_add_u32_e32 v76, 0xc400, v2
	v_add_u32_e32 v77, 0xc600, v2
	v_add_u32_e32 v82, 0xc800, v2
	v_add_u32_e32 v83, 0xca00, v2
	v_add_u32_e32 v80, 0xcc00, v2
	v_add_u32_e32 v81, 0xce00, v2
	v_add_u32_e32 v100, 0xa800, v2
	v_add_u32_e32 v101, 0xaa00, v2
	v_add_u32_e32 v102, 0xac00, v2
	v_add_u32_e32 v103, 0xae00, v2
	v_add_u32_e32 v123, 0xba00, v2
	v_add_u32_e32 v124, 0xbc00, v2
	v_add_u32_e32 v125, 0xbe00, v2
	v_or_b32_e32 v126, 0xc000, v2
	v_add_u32_e32 v127, 0xc200, v2
	v_or_b32_e32 v86, 0xd000, v2
	v_add_u32_e32 v87, 0xd200, v2
	v_add_u32_e32 v122, 0xd400, v2
	v_add_u32_e32 v121, 0xd600, v2
	v_add_u32_e32 v119, 0xd800, v2
	v_add_u32_e32 v120, 0xda00, v2
	v_add_u32_e32 v118, 0xdc00, v2
	v_add_u32_e32 v1, 0xde00, v2
	s_waitcnt vmcnt(10) lgkmcnt(1)
	v_pk_mul_f32 v[4:5], v[88:89], v[4:5]
	s_waitcnt vmcnt(8)
	v_pk_fma_f32 v[88:89], v[90:91], v[6:7], v[4:5]
	global_load_dword v104, v104, s[8:9]
	global_load_dword v105, v105, s[8:9]
	global_load_dword v106, v106, s[8:9]
	global_load_dword v107, v107, s[8:9]
	global_load_dword v108, v20, s[8:9]
	global_load_dword v109, v21, s[8:9]
	global_load_dword v110, v22, s[8:9]
	global_load_dword v111, v23, s[8:9]
	global_load_dword v112, v96, s[8:9]
	global_load_dword v113, v97, s[8:9]
	global_load_dword v114, v114, s[8:9]
	global_load_dword v115, v115, s[8:9]
	global_load_dword v6, v16, s[8:9]
	global_load_dword v7, v17, s[8:9]
	global_load_dword v4, v18, s[8:9]
	global_load_dword v5, v19, s[8:9]
	global_load_dword v18, v98, s[8:9]
	global_load_dword v19, v99, s[8:9]
	global_load_dword v16, v116, s[8:9]
	global_load_dword v17, v117, s[8:9]
	ds_read_b128 v[20:23], v3 offset:32
	s_waitcnt vmcnt(26) lgkmcnt(1)
	v_pk_fma_f32 v[8:9], v[92:93], v[8:9], v[88:89]
	ds_read_b128 v[88:91], v3 offset:48
	s_waitcnt vmcnt(24)
	v_pk_fma_f32 v[8:9], v[94:95], v[12:13], v[8:9]
	ds_read_b128 v[92:95], v3 offset:64
	ds_read_b128 v[96:99], v3 offset:80
	s_waitcnt vmcnt(22) lgkmcnt(3)
	v_pk_fma_f32 v[8:9], v[20:21], v[10:11], v[8:9]
	s_waitcnt vmcnt(20)
	v_pk_fma_f32 v[116:117], v[22:23], v[14:15], v[8:9]
	v_or_b32_e32 v8, 0x4000, v2
	global_load_dword v10, v8, s[8:9]
	global_load_dword v11, v31, s[8:9]
	global_load_dword v8, v38, s[8:9]
	global_load_dword v9, v39, s[8:9]
	global_load_dword v14, v30, s[8:9]
	global_load_dword v15, v29, s[8:9]
	global_load_dword v12, v27, s[8:9]
	global_load_dword v13, v28, s[8:9]
	global_load_dword v22, v26, s[8:9]
	global_load_dword v23, v25, s[8:9]
	global_load_dword v20, v24, s[8:9]
	global_load_dword v21, v47, s[8:9]
	global_load_dword v26, v48, s[8:9]
	global_load_dword v27, v49, s[8:9]
	global_load_dword v24, v50, s[8:9]
	global_load_dword v25, v51, s[8:9]
	global_load_dword v30, v52, s[8:9]
	global_load_dword v31, v53, s[8:9]
	global_load_dword v28, v54, s[8:9]
	global_load_dword v29, v55, s[8:9]
	global_load_dword v34, v34, s[8:9]
	global_load_dword v35, v35, s[8:9]
	global_load_dword v32, v32, s[8:9]
	global_load_dword v33, v33, s[8:9]
	global_load_dword v38, v36, s[8:9]
	global_load_dword v39, v37, s[8:9]
	global_load_dword v36, v62, s[8:9]
	global_load_dword v37, v63, s[8:9]
	global_load_dword v40, v40, s[8:9]
	global_load_dword v41, v41, s[8:9]
	global_load_dword v48, v46, s[8:9]
	global_load_dword v49, v45, s[8:9]
	global_load_dword v46, v68, s[8:9]
	global_load_dword v47, v69, s[8:9]
	global_load_dword v54, v70, s[8:9]
	global_load_dword v55, v71, s[8:9]
	global_load_dword v44, v44, s[8:9]
	global_load_dword v45, v73, s[8:9]
	global_load_dword v52, v74, s[8:9]
	global_load_dword v53, v75, s[8:9]
	global_load_dword v42, v42, s[8:9]
	global_load_dword v43, v43, s[8:9]
	global_load_dword v50, v78, s[8:9]
	global_load_dword v51, v79, s[8:9]
	global_load_dword v56, v56, s[8:9]
	global_load_dword v57, v57, s[8:9]
	global_load_dword v62, v61, s[8:9]
	global_load_dword v63, v60, s[8:9]
	global_load_dword v70, v84, s[8:9]
	global_load_dword v71, v85, s[8:9]
	global_load_dword v60, v59, s[8:9]
	global_load_dword v61, v58, s[8:9]
	global_load_dword v68, v100, s[8:9]
	global_load_dword v69, v101, s[8:9]
	global_load_dword v58, v102, s[8:9]
	global_load_dword v59, v103, s[8:9]
	global_load_dword v66, v66, s[8:9]
	global_load_dword v67, v67, s[8:9]
	global_load_dword v64, v64, s[8:9]
	global_load_dword v65, v65, s[8:9]
	global_load_dword v74, v72, s[8:9]
	global_load_dword v75, v123, s[8:9]
	global_load_dword v72, v124, s[8:9]
	global_load_dword v73, v125, s[8:9]
	global_load_dword v78, v126, s[8:9]
	global_load_dword v79, v127, s[8:9]
	global_load_dword v76, v76, s[8:9]
	global_load_dword v77, v77, s[8:9]
	global_load_dword v82, v82, s[8:9]
	global_load_dword v83, v83, s[8:9]
	global_load_dword v80, v80, s[8:9]
	global_load_dword v81, v81, s[8:9]
	ds_read_b128 v[100:103], v3 offset:96
	s_waitcnt vmcnt(62) lgkmcnt(3)
	v_pk_fma_f32 v[84:85], v[88:89], v[104:105], v[116:117]
	s_nop 0
	v_pk_fma_f32 v[84:85], v[90:91], v[106:107], v[84:85]
	s_waitcnt lgkmcnt(2)
	v_pk_fma_f32 v[84:85], v[92:93], v[108:109], v[84:85]
	v_add_u32_e32 v92, 0xe200, v2
	v_pk_fma_f32 v[84:85], v[94:95], v[110:111], v[84:85]
	v_add_u32_e32 v111, 0xee00, v2
	s_waitcnt lgkmcnt(1)
	v_pk_fma_f32 v[84:85], v[96:97], v[112:113], v[84:85]
	v_add_u32_e32 v93, 0xe400, v2
	v_pk_fma_f32 v[96:97], v[98:99], v[114:115], v[84:85]
	global_load_dword v86, v86, s[8:9]
	global_load_dword v87, v87, s[8:9]
	global_load_dword v84, v122, s[8:9]
	global_load_dword v85, v121, s[8:9]
	global_load_dword v88, v119, s[8:9]
	global_load_dword v89, v120, s[8:9]
	global_load_dword v90, v118, s[8:9]
	global_load_dword v91, v1, s[8:9]
	v_or_b32_e32 v1, 0xe000, v2
	v_add_u32_e32 v94, 0xe600, v2
	v_add_u32_e32 v95, 0xe800, v2
	v_add_u32_e32 v98, 0xea00, v2
	v_add_u32_e32 v99, 0xec00, v2
	global_load_dword v104, v1, s[8:9]
	global_load_dword v105, v92, s[8:9]
	global_load_dword v106, v93, s[8:9]
	global_load_dword v107, v94, s[8:9]
	global_load_dword v108, v95, s[8:9]
	global_load_dword v109, v98, s[8:9]
	global_load_dword v110, v99, s[8:9]
	global_load_dword v111, v111, s[8:9]
	v_or_b32_e32 v1, 0xf000, v2
	v_add_u32_e32 v92, 0xf200, v2
	v_add_u32_e32 v119, 0xfe00, v2
	v_add_u32_e32 v93, 0xf400, v2
	v_add_u32_e32 v94, 0xf600, v2
	v_add_u32_e32 v95, 0xf800, v2
	v_add_u32_e32 v98, 0xfa00, v2
	v_add_u32_e32 v99, 0xfc00, v2
	global_load_dword v112, v1, s[8:9]
	global_load_dword v113, v92, s[8:9]
	global_load_dword v114, v93, s[8:9]
	global_load_dword v115, v94, s[8:9]
	global_load_dword v116, v95, s[8:9]
	global_load_dword v117, v98, s[8:9]
	global_load_dword v118, v99, s[8:9]
	global_load_dword v119, v119, s[8:9]
	v_lshl_add_u32 v92, s2, 7, v0
	v_ashrrev_i32_e32 v93, 31, v92
	v_lshl_add_u64 v[92:93], v[92:93], 2, s[4:5]
	global_load_dword v1, v[92:93], off
	ds_read_b128 v[92:95], v3 offset:112
	s_waitcnt lgkmcnt(1)
	v_pk_fma_f32 v[6:7], v[100:101], v[6:7], v[96:97]
	ds_read_b128 v[96:99], v3 offset:128
	v_pk_fma_f32 v[4:5], v[102:103], v[4:5], v[6:7]
	s_load_dwordx2 s[4:5], s[0:1], 0x20
	s_waitcnt lgkmcnt(0)
	v_pk_fma_f32 v[18:19], v[92:93], v[18:19], v[4:5]
	ds_read_b128 v[4:7], v3 offset:144
	v_pk_fma_f32 v[16:17], v[94:95], v[16:17], v[18:19]
	s_nop 0
	v_pk_fma_f32 v[10:11], v[96:97], v[10:11], v[16:17]
	ds_read_b128 v[16:19], v3 offset:160
	v_pk_fma_f32 v[8:9], v[98:99], v[8:9], v[10:11]
	s_waitcnt lgkmcnt(1)
	v_pk_fma_f32 v[4:5], v[4:5], v[14:15], v[8:9]
	ds_read_b128 v[8:11], v3 offset:176
	v_pk_fma_f32 v[4:5], v[6:7], v[12:13], v[4:5]
	s_waitcnt lgkmcnt(1)
	v_pk_fma_f32 v[12:13], v[16:17], v[22:23], v[4:5]
	ds_read_b128 v[4:7], v3 offset:192
	s_waitcnt vmcnt(62)
	v_pk_fma_f32 v[12:13], v[18:19], v[20:21], v[12:13]
	s_waitcnt lgkmcnt(1)
	v_pk_fma_f32 v[8:9], v[8:9], v[26:27], v[12:13]
	ds_read_b128 v[12:15], v3 offset:208
	v_pk_fma_f32 v[8:9], v[10:11], v[24:25], v[8:9]
	s_waitcnt lgkmcnt(1)
	v_pk_fma_f32 v[4:5], v[4:5], v[30:31], v[8:9]
	ds_read_b128 v[8:11], v3 offset:224
	v_pk_fma_f32 v[4:5], v[6:7], v[28:29], v[4:5]
	s_waitcnt lgkmcnt(1)
	v_pk_fma_f32 v[12:13], v[12:13], v[34:35], v[4:5]
	ds_read_b128 v[4:7], v3 offset:240
	v_pk_fma_f32 v[12:13], v[14:15], v[32:33], v[12:13]
	s_waitcnt lgkmcnt(1)
	v_pk_fma_f32 v[8:9], v[8:9], v[38:39], v[12:13]
	ds_read_b128 v[12:15], v3 offset:256
	v_pk_fma_f32 v[8:9], v[10:11], v[36:37], v[8:9]
	s_waitcnt lgkmcnt(1)
	v_pk_fma_f32 v[4:5], v[4:5], v[40:41], v[8:9]
	s_nop 0
	v_pk_fma_f32 v[8:9], v[6:7], v[48:49], v[4:5]
	ds_read_b128 v[4:7], v3 offset:272
	s_waitcnt lgkmcnt(1)
	v_pk_fma_f32 v[8:9], v[12:13], v[46:47], v[8:9]
	s_waitcnt vmcnt(61)
	v_pk_fma_f32 v[16:17], v[14:15], v[54:55], v[8:9]
	ds_read_b128 v[8:11], v3 offset:288
	ds_read_b128 v[12:15], v3 offset:304
	s_waitcnt vmcnt(59) lgkmcnt(2)
	v_pk_fma_f32 v[4:5], v[4:5], v[44:45], v[16:17]
	s_waitcnt vmcnt(57)
	v_pk_fma_f32 v[20:21], v[6:7], v[52:53], v[4:5]
	ds_read_b128 v[4:7], v3 offset:320
	ds_read_b128 v[16:19], v3 offset:336
	s_waitcnt vmcnt(55) lgkmcnt(3)
	v_pk_fma_f32 v[8:9], v[8:9], v[42:43], v[20:21]
	s_waitcnt vmcnt(53)
	v_pk_fma_f32 v[8:9], v[10:11], v[50:51], v[8:9]
	s_waitcnt vmcnt(51) lgkmcnt(2)
	v_pk_fma_f32 v[8:9], v[12:13], v[56:57], v[8:9]
	s_waitcnt vmcnt(49)
	v_pk_fma_f32 v[8:9], v[14:15], v[62:63], v[8:9]
	s_waitcnt vmcnt(47) lgkmcnt(1)
	v_pk_fma_f32 v[4:5], v[4:5], v[70:71], v[8:9]
	ds_read_b128 v[8:11], v3 offset:352
	s_waitcnt vmcnt(45)
	v_pk_fma_f32 v[4:5], v[6:7], v[60:61], v[4:5]
	s_waitcnt vmcnt(43) lgkmcnt(1)
	v_pk_fma_f32 v[12:13], v[16:17], v[68:69], v[4:5]
	ds_read_b128 v[4:7], v3 offset:368
	s_waitcnt vmcnt(41)
	v_pk_fma_f32 v[12:13], v[18:19], v[58:59], v[12:13]
	s_waitcnt vmcnt(39) lgkmcnt(1)
	v_pk_fma_f32 v[8:9], v[8:9], v[66:67], v[12:13]
	ds_read_b128 v[12:15], v3 offset:384
	s_waitcnt vmcnt(37)
	v_pk_fma_f32 v[8:9], v[10:11], v[64:65], v[8:9]
	s_waitcnt vmcnt(35) lgkmcnt(1)
	v_pk_fma_f32 v[4:5], v[4:5], v[74:75], v[8:9]
	ds_read_b128 v[8:11], v3 offset:400
	s_waitcnt vmcnt(33)
	v_pk_fma_f32 v[4:5], v[6:7], v[72:73], v[4:5]
	s_waitcnt vmcnt(31) lgkmcnt(1)
	v_pk_fma_f32 v[12:13], v[12:13], v[78:79], v[4:5]
	ds_read_b128 v[4:7], v3 offset:416
	s_waitcnt vmcnt(29)
	v_pk_fma_f32 v[12:13], v[14:15], v[76:77], v[12:13]
	s_waitcnt vmcnt(27) lgkmcnt(1)
	v_pk_fma_f32 v[8:9], v[8:9], v[82:83], v[12:13]
	ds_read_b128 v[12:15], v3 offset:432
	s_waitcnt vmcnt(25)
	v_pk_fma_f32 v[8:9], v[10:11], v[80:81], v[8:9]
	s_waitcnt vmcnt(23) lgkmcnt(1)
	v_pk_fma_f32 v[4:5], v[4:5], v[86:87], v[8:9]
	ds_read_b128 v[8:11], v3 offset:448
	s_waitcnt vmcnt(21)
	v_pk_fma_f32 v[4:5], v[6:7], v[84:85], v[4:5]
	s_waitcnt vmcnt(19) lgkmcnt(1)
	v_pk_fma_f32 v[4:5], v[12:13], v[88:89], v[4:5]
	s_waitcnt vmcnt(17)
	v_pk_fma_f32 v[12:13], v[14:15], v[90:91], v[4:5]
	ds_read_b128 v[4:7], v3 offset:464
	s_waitcnt vmcnt(15) lgkmcnt(1)
	v_pk_fma_f32 v[8:9], v[8:9], v[104:105], v[12:13]
	ds_read_b128 v[12:15], v3 offset:480
	s_waitcnt vmcnt(13)
	v_pk_fma_f32 v[16:17], v[10:11], v[106:107], v[8:9]
	ds_read_b128 v[8:11], v3 offset:496
	s_waitcnt vmcnt(11) lgkmcnt(2)
	v_pk_fma_f32 v[4:5], v[4:5], v[108:109], v[16:17]
	s_waitcnt vmcnt(9)
	v_pk_fma_f32 v[4:5], v[6:7], v[110:111], v[4:5]
	s_waitcnt vmcnt(7) lgkmcnt(1)
	v_pk_fma_f32 v[4:5], v[12:13], v[112:113], v[4:5]
	s_waitcnt vmcnt(5)
	v_pk_fma_f32 v[4:5], v[14:15], v[114:115], v[4:5]
	s_waitcnt vmcnt(3) lgkmcnt(0)
	v_pk_fma_f32 v[4:5], v[8:9], v[116:117], v[4:5]
	s_waitcnt vmcnt(1)
	v_pk_fma_f32 v[4:5], v[10:11], v[118:119], v[4:5]
	s_nop 0
	v_add_f32_e32 v3, v4, v5
	s_waitcnt vmcnt(0)
	v_sub_f32_e32 v1, v1, v3
	ds_write_b32 v2, v1
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b32 v1, v2
	v_mbcnt_lo_u32_b32 v3, -1, 0
	v_mbcnt_hi_u32_b32 v3, -1, v3
	v_mov_b32_e32 v4, 0x80
	v_lshl_or_b32 v4, v3, 2, v4
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v2, v1, v1
	ds_bpermute_b32 v2, v4, v2
	v_and_b32_e32 v4, 63, v3
	v_cmp_gt_u32_e32 vcc, 48, v4
	s_waitcnt lgkmcnt(0)
	v_fmac_f32_e32 v2, v1, v1
	v_cndmask_b32_e64 v1, 0, 16, vcc
	v_add_lshl_u32 v1, v1, v3, 2
	ds_bpermute_b32 v1, v1, v2
	v_cmp_gt_u32_e32 vcc, 56, v4
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v1, v2, v1
	v_cndmask_b32_e64 v2, 0, 8, vcc
	v_add_lshl_u32 v2, v2, v3, 2
	ds_bpermute_b32 v2, v2, v1
	v_cmp_gt_u32_e32 vcc, 60, v4
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v1, v1, v2
	v_cndmask_b32_e64 v2, 0, 4, vcc
	v_add_lshl_u32 v2, v2, v3, 2
	ds_bpermute_b32 v2, v2, v1
	v_cmp_gt_u32_e32 vcc, 62, v4
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v1, v1, v2
	v_cndmask_b32_e64 v2, 0, 2, vcc
	v_add_lshl_u32 v2, v2, v3, 2
	ds_bpermute_b32 v2, v2, v1
	v_cmp_ne_u32_e32 vcc, 63, v4
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v2, v1, v2
	v_addc_co_u32_e32 v1, vcc, 0, v3, vcc
	v_lshlrev_b32_e32 v1, 2, v1
	ds_bpermute_b32 v3, v1, v2
	v_and_b32_e32 v1, 63, v0
	v_cmp_eq_u32_e32 vcc, 0, v1
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB5_3
; %bb.2:
	v_lshrrev_b32_e32 v1, 6, v0
	v_lshlrev_b32_e32 v1, 2, v1
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v2, v2, v3
	ds_write_b32 v1, v2 offset:512
.LBB5_3:
	s_or_b64 exec, exec, s[0:1]
	v_cmp_eq_u32_e64 s[0:1], 0, v0
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB5_5
; %bb.4:
	v_mov_b32_e32 v1, 0
	ds_read_b64 v[2:3], v1 offset:512
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v2, v3, v2
	ds_write_b32 v1, v2 offset:520
.LBB5_5:
	s_or_b64 exec, exec, s[6:7]
	v_lshlrev_b32_e32 v1, 9, v0
	s_waitcnt lgkmcnt(0)
	s_barrier
	global_load_dwordx4 v[2:5], v1, s[10:11] offset:48
	global_load_dwordx4 v[124:127], v1, s[10:11] offset:32
	global_load_dwordx4 v[10:13], v1, s[10:11] offset:80
	global_load_dwordx4 v[52:55], v1, s[10:11] offset:64
	global_load_dwordx4 v[24:27], v1, s[10:11] offset:144
	global_load_dwordx4 v[16:19], v1, s[10:11] offset:112
	global_load_dwordx4 v[44:47], v1, s[10:11] offset:96
	global_load_dwordx4 v[20:23], v1, s[10:11] offset:208
	global_load_dwordx4 v[104:107], v1, s[10:11] offset:352
	v_mov_b32_e32 v0, 0
	ds_read_b128 v[36:39], v0 offset:80
	ds_read_b128 v[48:51], v0 offset:64
	ds_read_b128 v[32:35], v0 offset:176
	s_mul_hi_i32 s3, s2, 20
	s_mul_i32 s2, s2, 20
	s_waitcnt lgkmcnt(2)
	scratch_store_dwordx4 off, v[36:39], off offset:80 ; 16-byte Folded Spill
	ds_read_b128 v[38:41], v0 offset:112
	s_waitcnt lgkmcnt(1)
	scratch_store_dwordx4 off, v[32:35], off offset:64 ; 16-byte Folded Spill
	s_add_u32 s2, s4, s2
	s_addc_u32 s3, s5, s3
	global_load_dwordx4 v[60:63], v1, s[10:11]
	s_waitcnt lgkmcnt(0)
	scratch_store_dwordx4 off, v[38:41], off offset:112 ; 16-byte Folded Spill
	ds_read_b128 v[40:43], v0 offset:144
	ds_read_b128 v[56:59], v0
	ds_read_b128 v[64:67], v0 offset:32
	ds_read_b128 v[100:103], v0 offset:352
	global_load_dwordx4 v[96:99], v1, s[10:11] offset:384
	ds_read_b128 v[92:95], v0 offset:384
	global_load_dwordx4 v[88:91], v1, s[10:11] offset:416
	ds_read_b128 v[76:79], v0 offset:448
	global_load_dwordx4 v[72:75], v1, s[10:11] offset:480
	ds_read_b128 v[112:115], v0 offset:256
	ds_read_b128 v[84:87], v0 offset:416
	global_load_dwordx4 v[80:83], v1, s[10:11] offset:448
	ds_read_b128 v[68:71], v0 offset:480
	s_waitcnt vmcnt(16)
	scratch_store_dwordx4 off, v[2:5], off offset:160 ; 16-byte Folded Spill
	global_load_dwordx4 v[4:7], v1, s[10:11] offset:16
	s_waitcnt vmcnt(16)
	scratch_store_dwordx4 off, v[10:13], off offset:192 ; 16-byte Folded Spill
	global_load_dwordx4 v[12:15], v1, s[10:11] offset:176
	s_waitcnt vmcnt(16)
	scratch_store_dwordx4 off, v[24:27], off offset:48 ; 16-byte Folded Spill
	ds_read_b128 v[26:29], v0 offset:16
	s_waitcnt vmcnt(16)
	scratch_store_dwordx4 off, v[16:19], off offset:96 ; 16-byte Folded Spill
	s_waitcnt vmcnt(15)
	scratch_store_dwordx4 off, v[20:23], off offset:16 ; 16-byte Folded Spill
	s_waitcnt lgkmcnt(0)
	scratch_store_dwordx4 off, v[26:29], off offset:208 ; 16-byte Folded Spill
	global_load_dwordx4 v[28:31], v1, s[10:11] offset:160
	s_waitcnt vmcnt(7)
	scratch_store_dwordx4 off, v[4:7], off offset:144 ; 16-byte Folded Spill
	ds_read_b128 v[6:9], v0 offset:48
	s_waitcnt vmcnt(6)
	scratch_store_dwordx4 off, v[12:15], off offset:32 ; 16-byte Folded Spill
	s_waitcnt lgkmcnt(0)
	v_pk_mul_f32 v[2:3], v[6:7], v[2:3]
	scratch_store_dwordx4 off, v[6:9], off offset:176 ; 16-byte Folded Spill
	v_pk_fma_f32 v[2:3], v[26:27], v[4:5], v[2:3]
	ds_read_b128 v[4:7], v0 offset:208
	v_pk_fma_f32 v[2:3], v[36:37], v[10:11], v[2:3]
	ds_read_b128 v[8:11], v0 offset:240
	v_pk_fma_f32 v[2:3], v[38:39], v[16:17], v[2:3]
	ds_read_b128 v[16:19], v0 offset:192
	v_pk_fma_f32 v[2:3], v[40:41], v[24:25], v[2:3]
	ds_read_b128 v[24:27], v0 offset:160
	v_pk_fma_f32 v[2:3], v[32:33], v[12:13], v[2:3]
	s_waitcnt lgkmcnt(3)
	scratch_store_dwordx4 off, v[4:7], off  ; 16-byte Folded Spill
	v_pk_fma_f32 v[2:3], v[4:5], v[20:21], v[2:3]
	global_load_dwordx4 v[4:7], v1, s[10:11] offset:240
	s_waitcnt lgkmcnt(2)
	scratch_store_dwordx4 off, v[8:11], off offset:480 ; 16-byte Folded Spill
	global_load_dwordx4 v[36:39], v1, s[10:11] offset:128
	global_load_dwordx4 v[20:23], v1, s[10:11] offset:192
	global_load_dwordx4 v[12:15], v1, s[10:11] offset:224
	scratch_store_dwordx4 off, v[40:43], off offset:128 ; 16-byte Folded Spill
	ds_read_b128 v[40:43], v0 offset:96
	ds_read_b128 v[32:35], v0 offset:128
	s_waitcnt vmcnt(5)
	scratch_store_dwordx4 off, v[4:7], off offset:496 ; 16-byte Folded Spill
	v_pk_fma_f32 v[2:3], v[8:9], v[4:5], v[2:3]
	global_load_dwordx4 v[4:7], v1, s[10:11] offset:272
	ds_read_b128 v[8:11], v0 offset:272
	s_waitcnt lgkmcnt(0)
	scratch_store_dwordx4 off, v[8:11], off offset:448 ; 16-byte Folded Spill
	s_waitcnt vmcnt(1)
	scratch_store_dwordx4 off, v[4:7], off offset:464 ; 16-byte Folded Spill
	v_pk_fma_f32 v[2:3], v[8:9], v[4:5], v[2:3]
	global_load_dwordx4 v[4:7], v1, s[10:11] offset:304
	ds_read_b128 v[8:11], v0 offset:304
	s_waitcnt lgkmcnt(0)
	scratch_store_dwordx4 off, v[8:11], off offset:416 ; 16-byte Folded Spill
	s_waitcnt vmcnt(1)
	scratch_store_dwordx4 off, v[4:7], off offset:432 ; 16-byte Folded Spill
	v_pk_fma_f32 v[2:3], v[8:9], v[4:5], v[2:3]
	global_load_dwordx4 v[4:7], v1, s[10:11] offset:336
	ds_read_b128 v[8:11], v0 offset:336
	s_waitcnt lgkmcnt(0)
	scratch_store_dwordx4 off, v[8:11], off offset:384 ; 16-byte Folded Spill
	s_waitcnt vmcnt(1)
	scratch_store_dwordx4 off, v[4:7], off offset:400 ; 16-byte Folded Spill
	v_pk_fma_f32 v[2:3], v[8:9], v[4:5], v[2:3]
	global_load_dwordx4 v[4:7], v1, s[10:11] offset:368
	ds_read_b128 v[8:11], v0 offset:368
	s_waitcnt lgkmcnt(0)
	scratch_store_dwordx4 off, v[8:11], off offset:352 ; 16-byte Folded Spill
	s_waitcnt vmcnt(1)
	scratch_store_dwordx4 off, v[4:7], off offset:368 ; 16-byte Folded Spill
	v_pk_fma_f32 v[2:3], v[8:9], v[4:5], v[2:3]
	global_load_dwordx4 v[4:7], v1, s[10:11] offset:400
	ds_read_b128 v[8:11], v0 offset:400
	s_waitcnt lgkmcnt(0)
	scratch_store_dwordx4 off, v[8:11], off offset:320 ; 16-byte Folded Spill
	s_waitcnt vmcnt(1)
	scratch_store_dwordx4 off, v[4:7], off offset:336 ; 16-byte Folded Spill
	v_pk_fma_f32 v[2:3], v[8:9], v[4:5], v[2:3]
	global_load_dwordx4 v[4:7], v1, s[10:11] offset:432
	ds_read_b128 v[8:11], v0 offset:432
	s_waitcnt lgkmcnt(0)
	scratch_store_dwordx4 off, v[8:11], off offset:288 ; 16-byte Folded Spill
	s_waitcnt vmcnt(1)
	scratch_store_dwordx4 off, v[4:7], off offset:304 ; 16-byte Folded Spill
	v_pk_fma_f32 v[2:3], v[8:9], v[4:5], v[2:3]
	global_load_dwordx4 v[4:7], v1, s[10:11] offset:464
	ds_read_b128 v[8:11], v0 offset:464
	s_waitcnt lgkmcnt(0)
	scratch_store_dwordx4 off, v[8:11], off offset:256 ; 16-byte Folded Spill
	s_waitcnt vmcnt(1)
	scratch_store_dwordx4 off, v[4:7], off offset:272 ; 16-byte Folded Spill
	v_pk_fma_f32 v[2:3], v[8:9], v[4:5], v[2:3]
	global_load_dwordx4 v[4:7], v1, s[10:11] offset:496
	ds_read_b128 v[8:11], v0 offset:496
	s_waitcnt lgkmcnt(0)
	scratch_store_dwordx4 off, v[8:11], off offset:224 ; 16-byte Folded Spill
	s_waitcnt vmcnt(1)
	scratch_store_dwordx4 off, v[4:7], off offset:240 ; 16-byte Folded Spill
	v_pk_fma_f32 v[2:3], v[8:9], v[4:5], v[2:3]
	global_load_dwordx4 v[4:7], v1, s[10:11] offset:256
	v_pk_fma_f32 v[2:3], v[64:65], v[124:125], v[2:3]
	ds_read_b128 v[8:11], v0 offset:224
	v_pk_fma_f32 v[2:3], v[56:57], v[60:61], v[2:3]
	global_load_dwordx4 v[122:125], v1, s[10:11] offset:320
	v_pk_fma_f32 v[2:3], v[48:49], v[52:53], v[2:3]
	s_nop 0
	v_pk_fma_f32 v[2:3], v[40:41], v[44:45], v[2:3]
	s_nop 0
	v_pk_fma_f32 v[2:3], v[32:33], v[36:37], v[2:3]
	s_nop 0
	v_pk_fma_f32 v[2:3], v[24:25], v[28:29], v[2:3]
	s_nop 0
	v_pk_fma_f32 v[2:3], v[16:17], v[20:21], v[2:3]
	s_waitcnt lgkmcnt(0)
	v_pk_fma_f32 v[2:3], v[8:9], v[12:13], v[2:3]
	s_waitcnt vmcnt(1)
	v_pk_fma_f32 v[8:9], v[112:113], v[4:5], v[2:3]
	global_load_dwordx4 v[2:5], v1, s[10:11] offset:288
	ds_read_b128 v[110:113], v0 offset:288
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_pk_fma_f32 v[2:3], v[110:111], v[2:3], v[8:9]
	ds_read_b128 v[108:111], v0 offset:320
	s_waitcnt lgkmcnt(0)
	v_pk_fma_f32 v[2:3], v[108:109], v[122:123], v[2:3]
	scratch_load_dwordx4 v[120:123], off, off offset:160 ; 16-byte Folded Reload
	scratch_load_dwordx4 v[116:119], off, off offset:176 ; 16-byte Folded Reload
	v_pk_fma_f32 v[2:3], v[100:101], v[104:105], v[2:3]
	s_nop 0
	v_pk_fma_f32 v[2:3], v[92:93], v[96:97], v[2:3]
	s_nop 0
	v_pk_fma_f32 v[2:3], v[84:85], v[88:89], v[2:3]
	s_nop 0
	v_pk_fma_f32 v[2:3], v[76:77], v[80:81], v[2:3]
	s_nop 0
	v_pk_fma_f32 v[2:3], v[68:69], v[72:73], v[2:3]
	s_waitcnt vmcnt(0)
	v_pk_fma_f32 v[2:3], v[118:119], v[122:123], v[2:3]
	scratch_load_dwordx4 v[120:123], off, off offset:144 ; 16-byte Folded Reload
	scratch_load_dwordx4 v[116:119], off, off offset:208 ; 16-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_pk_fma_f32 v[2:3], v[118:119], v[122:123], v[2:3]
	scratch_load_dwordx4 v[118:121], off, off offset:80 ; 16-byte Folded Reload
	scratch_load_dwordx4 v[116:119], off, off offset:192 ; 16-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_pk_fma_f32 v[2:3], v[120:121], v[118:119], v[2:3]
	scratch_load_dwordx4 v[118:121], off, off offset:96 ; 16-byte Folded Reload
	scratch_load_dwordx4 v[116:119], off, off offset:112 ; 16-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_pk_fma_f32 v[2:3], v[118:119], v[120:121], v[2:3]
	scratch_load_dwordx4 v[120:123], off, off offset:48 ; 16-byte Folded Reload
	scratch_load_dwordx4 v[116:119], off, off offset:128 ; 16-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_pk_fma_f32 v[2:3], v[118:119], v[122:123], v[2:3]
	scratch_load_dwordx4 v[120:123], off, off offset:32 ; 16-byte Folded Reload
	scratch_load_dwordx4 v[116:119], off, off offset:64 ; 16-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_pk_fma_f32 v[2:3], v[118:119], v[122:123], v[2:3]
	scratch_load_dwordx4 v[116:119], off, off ; 16-byte Folded Reload
	scratch_load_dwordx4 v[120:123], off, off offset:16 ; 16-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_pk_fma_f32 v[2:3], v[118:119], v[122:123], v[2:3]
	scratch_load_dwordx4 v[116:119], off, off offset:480 ; 16-byte Folded Reload
	scratch_load_dwordx4 v[120:123], off, off offset:496 ; 16-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_pk_fma_f32 v[2:3], v[118:119], v[122:123], v[2:3]
	scratch_load_dwordx4 v[116:119], off, off offset:448 ; 16-byte Folded Reload
	scratch_load_dwordx4 v[120:123], off, off offset:464 ; 16-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_pk_fma_f32 v[2:3], v[118:119], v[122:123], v[2:3]
	scratch_load_dwordx4 v[116:119], off, off offset:416 ; 16-byte Folded Reload
	scratch_load_dwordx4 v[120:123], off, off offset:432 ; 16-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_pk_fma_f32 v[2:3], v[118:119], v[122:123], v[2:3]
	scratch_load_dwordx4 v[116:119], off, off offset:384 ; 16-byte Folded Reload
	scratch_load_dwordx4 v[120:123], off, off offset:400 ; 16-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_pk_fma_f32 v[2:3], v[118:119], v[122:123], v[2:3]
	scratch_load_dwordx4 v[116:119], off, off offset:352 ; 16-byte Folded Reload
	scratch_load_dwordx4 v[120:123], off, off offset:368 ; 16-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_pk_fma_f32 v[2:3], v[118:119], v[122:123], v[2:3]
	scratch_load_dwordx4 v[116:119], off, off offset:320 ; 16-byte Folded Reload
	scratch_load_dwordx4 v[120:123], off, off offset:336 ; 16-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_pk_fma_f32 v[2:3], v[118:119], v[122:123], v[2:3]
	scratch_load_dwordx4 v[116:119], off, off offset:288 ; 16-byte Folded Reload
	scratch_load_dwordx4 v[120:123], off, off offset:304 ; 16-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_pk_fma_f32 v[2:3], v[118:119], v[122:123], v[2:3]
	scratch_load_dwordx4 v[116:119], off, off offset:256 ; 16-byte Folded Reload
	scratch_load_dwordx4 v[120:123], off, off offset:272 ; 16-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_pk_fma_f32 v[2:3], v[118:119], v[122:123], v[2:3]
	scratch_load_dwordx4 v[116:119], off, off offset:224 ; 16-byte Folded Reload
	scratch_load_dwordx4 v[120:123], off, off offset:240 ; 16-byte Folded Reload
	ds_read_b32 v0, v0 offset:520
	s_waitcnt vmcnt(0)
	v_pk_fma_f32 v[2:3], v[118:119], v[122:123], v[2:3]
	s_nop 0
	v_pk_fma_f32 v[2:3], v[66:67], v[126:127], v[2:3]
	s_nop 0
	v_pk_fma_f32 v[2:3], v[58:59], v[62:63], v[2:3]
	s_nop 0
	v_pk_fma_f32 v[2:3], v[50:51], v[54:55], v[2:3]
	s_nop 0
	v_pk_fma_f32 v[2:3], v[42:43], v[46:47], v[2:3]
	s_nop 0
	v_pk_fma_f32 v[2:3], v[34:35], v[38:39], v[2:3]
	s_nop 0
	v_pk_fma_f32 v[2:3], v[26:27], v[30:31], v[2:3]
	s_nop 0
	v_pk_fma_f32 v[2:3], v[18:19], v[22:23], v[2:3]
	s_nop 0
	v_pk_fma_f32 v[2:3], v[10:11], v[14:15], v[2:3]
	s_nop 0
	v_pk_fma_f32 v[2:3], v[114:115], v[6:7], v[2:3]
	s_nop 0
	v_pk_fma_f32 v[2:3], v[112:113], v[4:5], v[2:3]
	s_nop 0
	v_pk_fma_f32 v[2:3], v[110:111], v[124:125], v[2:3]
	s_nop 0
	v_pk_fma_f32 v[2:3], v[102:103], v[106:107], v[2:3]
	s_nop 0
	v_pk_fma_f32 v[2:3], v[94:95], v[98:99], v[2:3]
	s_nop 0
	v_pk_fma_f32 v[2:3], v[86:87], v[90:91], v[2:3]
	s_nop 0
	v_pk_fma_f32 v[2:3], v[78:79], v[82:83], v[2:3]
	s_nop 0
	v_pk_fma_f32 v[2:3], v[70:71], v[74:75], v[2:3]
	s_nop 0
	v_add_f32_e32 v1, v2, v3
	v_cmp_lt_f32_e64 s[6:7], 0, v1
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB5_7
; %bb.6:
	scratch_load_dword v1, off, off offset:512 ; 4-byte Folded Reload
	v_mov_b64_e32 v[2:3], s[6:7]
	s_waitcnt vmcnt(0)
	global_store_dwordx2 v1, v[2:3], s[2:3]
.LBB5_7:
	s_or_b64 exec, exec, s[4:5]
	s_and_saveexec_b64 s[4:5], s[0:1]
	s_cbranch_execz .LBB5_9
; %bb.8:
	s_waitcnt lgkmcnt(0)
	v_sqrt_f32_e32 v0, v0
	v_mov_b32_e32 v1, 0
	global_store_dword v1, v0, s[2:3] offset:16
.LBB5_9:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z14tqm_qjl_kernelPKfPK16block_tq3_mi300xS0_S0_P16block_qjl_mi300xi
		.amdhsa_group_segment_fixed_size 524
		.amdhsa_private_segment_fixed_size 520
		.amdhsa_kernarg_size 44
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 0
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 128
		.amdhsa_next_free_sgpr 13
		.amdhsa_accum_offset 128
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_tg_split 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end5:
	.size	_Z14tqm_qjl_kernelPKfPK16block_tq3_mi300xS0_S0_P16block_qjl_mi300xi, .Lfunc_end5-_Z14tqm_qjl_kernelPKfPK16block_tq3_mi300xS0_S0_P16block_qjl_mi300xi
                                        ; -- End function
	.set _Z14tqm_qjl_kernelPKfPK16block_tq3_mi300xS0_S0_P16block_qjl_mi300xi.num_vgpr, 128
	.set _Z14tqm_qjl_kernelPKfPK16block_tq3_mi300xS0_S0_P16block_qjl_mi300xi.num_agpr, 0
	.set _Z14tqm_qjl_kernelPKfPK16block_tq3_mi300xS0_S0_P16block_qjl_mi300xi.numbered_sgpr, 13
	.set _Z14tqm_qjl_kernelPKfPK16block_tq3_mi300xS0_S0_P16block_qjl_mi300xi.num_named_barrier, 0
	.set _Z14tqm_qjl_kernelPKfPK16block_tq3_mi300xS0_S0_P16block_qjl_mi300xi.private_seg_size, 520
	.set _Z14tqm_qjl_kernelPKfPK16block_tq3_mi300xS0_S0_P16block_qjl_mi300xi.uses_vcc, 1
	.set _Z14tqm_qjl_kernelPKfPK16block_tq3_mi300xS0_S0_P16block_qjl_mi300xi.uses_flat_scratch, 0
	.set _Z14tqm_qjl_kernelPKfPK16block_tq3_mi300xS0_S0_P16block_qjl_mi300xi.has_dyn_sized_stack, 0
	.set _Z14tqm_qjl_kernelPKfPK16block_tq3_mi300xS0_S0_P16block_qjl_mi300xi.has_recursion, 0
	.set _Z14tqm_qjl_kernelPKfPK16block_tq3_mi300xS0_S0_P16block_qjl_mi300xi.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 5612
; TotalNumSgprs: 19
; NumVgprs: 128
; NumAgprs: 0
; TotalNumVgprs: 128
; ScratchSize: 520
; MemoryBound: 1
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 524 bytes/workgroup (compile time only)
; SGPRBlocks: 2
; VGPRBlocks: 15
; NumSGPRsForWavesPerEU: 19
; NumVGPRsForWavesPerEU: 128
; AccumOffset: 128
; Occupancy: 4
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 1
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 31
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.text
	.protected	d_cb3                   ; @d_cb3
	.type	d_cb3,@object
	.section	.rodata,"a",@progbits
	.globl	d_cb3
	.p2align	4, 0x0
d_cb3:
	.zero	32
	.size	d_cb3, 32

	.protected	d_cb4                   ; @d_cb4
	.type	d_cb4,@object
	.globl	d_cb4
	.p2align	4, 0x0
d_cb4:
	.zero	64
	.size	d_cb4, 64

	.protected	d_cb2                   ; @d_cb2
	.type	d_cb2,@object
	.globl	d_cb2
	.p2align	4, 0x0
d_cb2:
	.zero	16
	.size	d_cb2, 16

	.type	__hip_cuid_81ebb66926e4b9a5,@object ; @__hip_cuid_81ebb66926e4b9a5
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_81ebb66926e4b9a5
__hip_cuid_81ebb66926e4b9a5:
	.byte	0                               ; 0x0
	.size	__hip_cuid_81ebb66926e4b9a5, 1

	.ident	"AMD clang version 22.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.2.0 26014 7b800a19466229b8479a78de19143dc33c3ab9b5)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym d_cb3
	.addrsig_sym d_cb4
	.addrsig_sym d_cb2
	.addrsig_sym __hip_cuid_81ebb66926e4b9a5
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     0
    .args:
      - .actual_access:  read_only
        .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  write_only
        .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .offset:         24
        .size:           4
        .value_kind:     by_value
    .group_segment_fixed_size: 524
    .kernarg_segment_align: 8
    .kernarg_segment_size: 28
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _Z23tqm_quantize_kernel_tq3PKfS0_P16block_tq3_mi300xi
    .private_segment_fixed_size: 644
    .sgpr_count:     22
    .sgpr_spill_count: 0
    .symbol:         _Z23tqm_quantize_kernel_tq3PKfS0_P16block_tq3_mi300xi.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     128
    .vgpr_spill_count: 160
    .wavefront_size: 64
  - .agpr_count:     0
    .args:
      - .actual_access:  read_only
        .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  write_only
        .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .offset:         24
        .size:           4
        .value_kind:     by_value
    .group_segment_fixed_size: 516
    .kernarg_segment_align: 8
    .kernarg_segment_size: 28
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _Z25tqm_dequantize_kernel_tq3PK16block_tq3_mi300xPKfPfi
    .private_segment_fixed_size: 0
    .sgpr_count:     16
    .sgpr_spill_count: 0
    .symbol:         _Z25tqm_dequantize_kernel_tq3PK16block_tq3_mi300xPKfPfi.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     75
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .agpr_count:     0
    .args:
      - .actual_access:  read_only
        .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  write_only
        .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .offset:         24
        .size:           4
        .value_kind:     by_value
      - .offset:         28
        .size:           4
        .value_kind:     by_value
    .group_segment_fixed_size: 12
    .kernarg_segment_align: 8
    .kernarg_segment_size: 32
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _Z24tqm_fused_dot_kernel_tq3PKfPK16block_tq3_mi300xPfii
    .private_segment_fixed_size: 0
    .sgpr_count:     20
    .sgpr_spill_count: 0
    .symbol:         _Z24tqm_fused_dot_kernel_tq3PKfPK16block_tq3_mi300xPfii.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     12
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .agpr_count:     0
    .args:
      - .actual_access:  read_only
        .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  write_only
        .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .offset:         24
        .size:           4
        .value_kind:     by_value
    .group_segment_fixed_size: 524
    .kernarg_segment_align: 8
    .kernarg_segment_size: 28
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _Z23tqm_quantize_kernel_tq4PKfS0_P16block_tq4_mi300xi
    .private_segment_fixed_size: 644
    .sgpr_count:     30
    .sgpr_spill_count: 0
    .symbol:         _Z23tqm_quantize_kernel_tq4PKfS0_P16block_tq4_mi300xi.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     128
    .vgpr_spill_count: 160
    .wavefront_size: 64
  - .agpr_count:     0
    .args:
      - .actual_access:  read_only
        .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  write_only
        .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .offset:         24
        .size:           4
        .value_kind:     by_value
    .group_segment_fixed_size: 516
    .kernarg_segment_align: 8
    .kernarg_segment_size: 28
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _Z25tqm_dequantize_kernel_tq4PK16block_tq4_mi300xPKfPfi
    .private_segment_fixed_size: 0
    .sgpr_count:     16
    .sgpr_spill_count: 0
    .symbol:         _Z25tqm_dequantize_kernel_tq4PK16block_tq4_mi300xPKfPfi.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     75
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .agpr_count:     0
    .args:
      - .actual_access:  read_only
        .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  write_only
        .address_space:  global
        .offset:         32
        .size:           8
        .value_kind:     global_buffer
      - .offset:         40
        .size:           4
        .value_kind:     by_value
    .group_segment_fixed_size: 524
    .kernarg_segment_align: 8
    .kernarg_segment_size: 44
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _Z14tqm_qjl_kernelPKfPK16block_tq3_mi300xS0_S0_P16block_qjl_mi300xi
    .private_segment_fixed_size: 520
    .sgpr_count:     19
    .sgpr_spill_count: 0
    .symbol:         _Z14tqm_qjl_kernelPKfPK16block_tq3_mi300xS0_S0_P16block_qjl_mi300xi.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     128
    .vgpr_spill_count: 129
    .wavefront_size: 64
amdhsa.target:   'amdgcn-amd-amdhsa--gfx942:sramecc+:xnack-'
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
