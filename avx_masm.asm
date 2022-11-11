;extern "C" void FindRow_ (double* Input1, double* Input2, double* Output);
;Fills the row of 1024 doubles starting from Output

_Helper macro disp

    ;.......................................
    ;Broadcast the first element of first matrix into y4 register
    vbroadcastsd ymm4, real8 ptr [rcx+8*disp]

    ;Get the row of second matrix into y5,y6,y7,y8
    ;vmovapd ymm5, ymmword ptr [rdx+8192*disp]
    ;vmovapd ymm6, ymmword ptr [rdx+32+8192*disp]
    ;vmovapd ymm7, ymmword ptr [rdx+64+8192*disp]
    ;vmovapd ymm8, ymmword ptr [rdx+96+8192*disp]

    ;Fuse time
    vfmadd231pd ymm0, ymm4, ymmword ptr [rdx+8192*disp]
    vfmadd231pd ymm1, ymm4, ymmword ptr [rdx+32+8192*disp]
    vfmadd231pd ymm2, ymm4, ymmword ptr [rdx+64+8192*disp]
    vfmadd231pd ymm3, ymm4, ymmword ptr [rdx+96+8192*disp]
    ;.......................................

endm

.code
FindRow_ proc

    ;rcx = Input1
	;rdx = Input2
    ;r8 = Output
    ;r9 = N

    ;Get the output matrix row into y0,y1,y2,y3 registers
    vmovapd ymm0, ymmword ptr [r8]
    vmovapd ymm1, ymmword ptr [r8+32]
    vmovapd ymm2, ymmword ptr [r8+64]
    vmovapd ymm3, ymmword ptr [r8+96]

    ;.......................................
    _Helper 0
    _Helper 1
    _Helper 2
    _Helper 3
    _Helper 4
    _Helper 5
    _Helper 6
    _Helper 7
    _Helper 8
    _Helper 9
    _Helper 10
    _Helper 11
    _Helper 12
    _Helper 13
    _Helper 14
    _Helper 15
    ;.......................................

    ;Send the values back into output row
    vmovapd ymmword ptr [r8], ymm0
    vmovapd ymmword ptr [r8+32], ymm1
    vmovapd ymmword ptr [r8+64], ymm2
    vmovapd ymmword ptr [r8+96], ymm3

	ret

FindRow_ endp
	end