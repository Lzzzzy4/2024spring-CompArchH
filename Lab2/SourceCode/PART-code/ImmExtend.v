`timescale 1ns / 1ps
//  功能说明
    //  立即数拓展，将指令中的立即数部分拓展为完整立即数
// 输入
    // Inst              指令的[31:7]
    // ImmType           立即数类型
// 输出
    // imm               补全的立即数
// 实验要求
    // 补全模块


`include "Parameters.v"   
module ImmExtend(
    input wire [31:7] inst,
    input wire [2:0] imm_type,
    output reg [31:0] imm
    );

    always@(*)
    begin
        case(imm_type)
            `RTYPE: imm = 0;
            `ITYPE: imm = {{20{inst[31]}}, inst[31:20]};
            // TODO: complete left part
            // Parameters.v defines all immediate type
            `STYPE: imm = {{20{inst[31]}}, inst[31:25], inst[11:7]};
            `BTYPE: imm = {{19{inst[31]}}, inst[31], inst[7], inst[30:25], inst[11:8], 1'b0};
            `UTYPE: imm = {inst[31:12], 12'd0};
            `JTYPE: imm = {{11{inst[31]}}, inst[31], inst[19:12], inst[20], inst[30:21], 1'b0};

            /* FIXM: Write your code here... */

            default: imm = 32'b0;
        endcase
    end
endmodule
