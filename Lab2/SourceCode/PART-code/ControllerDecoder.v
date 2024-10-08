`timescale 1ns / 1ps
//  功能说明
    //  对指令进行译码，将其翻译成控制信号，传输给各个部件
// 输入
    // Inst              待译码指令
// 输出
    // jal               jal跳转指令
    // jalr              jalr跳转指令
    // op1_src           0表示ALU操作数1来自寄存器，1表示来自PC-4
    // op2_src           ALU的第二个操作数来源。为1时，op2选择imm，为0时，op2选择reg2
    // ALU_func          ALU执行的运算类型
    // br_type           branch的判断条件，可以是不进行branch
    // load_npc          写回寄存器的值的来源（PC或者ALU计算结果）, load_npc == 1时选择PC
    // wb_select         写回寄存器的值的来源（Cache内容或者ALU计算结果），wb_select == 1时选择cache内容
    // load_type         load类型
    // reg_write_en      通用寄存器写使能，reg_write_en == 1表示需要写回reg
    // cache_write_en    按字节写入data cache
    // imm_type          指令中立即数类型
    // CSR_write_en
    // CSR_zimm_or_reg
// 实验要求
    // 补全模块


`include "Parameters.v"   
module ControllerDecoder(
    input wire [31:0] inst,
    output reg jal,
    output reg jalr,
    output reg op1_src, op2_src,
    output reg [3:0] ALU_func,
    output reg [2:0] br_type,
    output reg load_npc,
    output reg wb_select,
    output reg [2:0] load_type,
    output reg reg_write_en,
    output reg [3:0] cache_write_en,
    output reg [2:0] imm_type,
    // CSR signals
    output reg CSR_write_en,
    output reg CSR_zimm_or_reg
    );

    // TODO: Complete this module

    wire [6:0] opcode, funct7;
    wire [2:0] funct3;
    reg [31:0]test;

    assign opcode = inst[6:0];
    assign funct7 = inst[31:25];
    assign funct3 = inst[14:12];

    wire [4:0] rs1;
    assign rs1 = inst[19:15];
    wire [4:0] rs2;
    assign rs2 = inst[24:20];
    wire [4:0] rd;
    assign rd = inst[11:7];

    always @ (*) begin
        jal = 0;
        jalr = 0;
        op1_src = 0;
        op2_src = 0;
        ALU_func = 0;
        br_type = 0;
        load_npc = 0;
        wb_select = 0;
        load_type = 0;
        reg_write_en = 0;
        cache_write_en = 0;
        imm_type = 0;
        CSR_write_en = 0;
        CSR_zimm_or_reg = 0;
        if (opcode == `U_LUI) begin
            test = 32'd1;
            jal = 0;
            jalr = 0;
            op1_src = 0;
            op2_src = `IMM;
            ALU_func = `LUI;
            br_type = 0;
            load_npc = 0;
            wb_select = 0;
            load_type = 0;
            reg_write_en = 1;
            cache_write_en = 0;
            imm_type = `UTYPE;
            CSR_write_en = 0;
            CSR_zimm_or_reg = 0;
        end
        else if (opcode == `U_AUIPC) begin
            test = 32'd2;
            jal = 0;
            jalr = 0;
            op1_src = `PC_4;
            op2_src = `IMM;
            ALU_func = `ADD;
            br_type = 0;
            load_npc = 0;
            wb_select = 0;
            load_type = 0;
            reg_write_en = 1;
            cache_write_en = 0;
            imm_type = `UTYPE;
            CSR_write_en = 0;
            CSR_zimm_or_reg = 0;
        end
        else if (opcode == `J_JAL) begin
            test = 32'd3;
            jal = 1;
            jalr = 0;
            op1_src = 0;
            op2_src = 0;
            ALU_func = 0;
            br_type = 0;
            load_npc = 1; // load npc to reg
            wb_select = 0;
            load_type = 0;
            reg_write_en = 1;
            cache_write_en = 0;
            imm_type = `JTYPE;
            CSR_write_en = 0;
            CSR_zimm_or_reg = 0;
        end
        else if (opcode == `J_JALR) begin
            test = 32'd4;
            jal = 0;
            jalr = 1;
            op1_src = `REG1;
            op2_src = `IMM;
            ALU_func = `ADD;
            br_type = 0;
            load_npc = 1;
            wb_select = 0;
            load_type = 0;
            reg_write_en = 1;
            cache_write_en = 0;
            imm_type = `ITYPE;
            CSR_write_en = 0;
            CSR_zimm_or_reg = 0;
        end
        else if (opcode == `B_TYPE) begin
            test = 32'd5;
            jal = 0;
            jalr = 0;
            op1_src = `REG1;
            op2_src = `REG2;
            ALU_func = 0;
            if (funct3 == 3'b000) br_type = `BEQ;
            else if (funct3 == 3'b001) br_type = `BNE;
            else if (funct3 == 3'b100) br_type = `BLT;
            else if (funct3 == 3'b101) br_type = `BGE;
            else if (funct3 == 3'b110) br_type = `BLTU;
            else if (funct3 == 3'b111) br_type = `BGEU;
            load_npc = 0;
            wb_select = 0;
            load_type = 0;
            reg_write_en = 0;
            cache_write_en = 0;
            imm_type = `BTYPE;
            CSR_write_en = 0;
            CSR_zimm_or_reg = 0;
        end
        else if (opcode == `I_LOAD) begin
            test = 32'd6;
            jal = 0;
            jalr = 0;
            op1_src = 0;
            op2_src = 1;
            ALU_func = `ADD;
            br_type = 0;
            load_npc = 0;
            wb_select = 1;
            
            reg_write_en = 1;
            // cache_write_en = 0;
            cache_write_en = 4'b0010; // means cache read
            imm_type = `ITYPE;
            CSR_write_en = 0;
            CSR_zimm_or_reg = 0;
            if (funct3 == `I_LB) begin
                load_type = `LB;
            end
            else if (funct3 == `I_LH) begin
                load_type = `LH;
            end
            else if (funct3 == `I_LW) begin
                load_type = `LW;
            end
            else if (funct3 == `I_LBU) begin
                load_type = `LBU;
            end
            else if (funct3 == `I_LHU) begin
                load_type = `LHU;
            end
            else  begin
                reg_write_en = 0;
                load_type = `NOREGWRITE;
            end
        end

        /* FIXM: Write your code here... */
        else if (opcode == `I_ARI) begin
            test = 32'd7;
            jal = 0;
            jalr = 0;
            op1_src = `REG1;
            op2_src = `IMM;
            if (funct3 == `I_ADDI)          ALU_func = `ADD;
            else if (funct3 == `I_SLTI)     ALU_func = `SLT;
            else if (funct3 == `I_SLTIU)    ALU_func = `SLTU;
            else if (funct3 == `I_XORI)     ALU_func = `XOR;
            else if (funct3 == `I_ORI)      ALU_func = `OR;
            else if (funct3 == `I_ANDI)     ALU_func = `AND;
            else if (funct3 == `I_SLLI)     ALU_func = `SLL;
            else if (funct3 == `I_SR) begin
                if (funct7 == `I_SRAI)      ALU_func = `SRA;
                else if (funct7 == `I_SRLI) ALU_func = `SRL;
            end
            br_type = 0;
            load_npc = 0;
            wb_select = 0;
            load_type = 0;
            reg_write_en = 1;
            cache_write_en = 0;
            imm_type = `ITYPE;
            CSR_write_en = 0;
            CSR_zimm_or_reg = 0;
        end
        else if (opcode == `S_TYPE) begin
            test = 32'd8;
            jal = 0;
            jalr = 0;
            op1_src = `REG1;
            op2_src = `IMM;
            ALU_func = `ADD;
            br_type = 0;
            load_npc = 0;
            wb_select = 0;
            load_type = 0;
            reg_write_en = 0;

            // if (funct3 == `S_SB) cache_write_en = 4'b0001;
            // else if (funct3 == `S_SH) cache_write_en = 4'b0011;
            // else if (funct3 == `S_SW) cache_write_en = 4'b1111;
            cache_write_en = 4'b0001; // means cache write

            imm_type = `STYPE;
            CSR_write_en = 0;
            CSR_zimm_or_reg = 0;
        end
        else if (opcode == `R_TYPE) begin
            test = 32'd9;
            jal = 0;
            jalr = 0;
            op1_src = `REG1;
            op2_src = `REG2;
            if (funct3 == `R_AS) begin
                if (funct7 == `R_ADD) ALU_func = `ADD;
                else if (funct7 == `R_SUB) ALU_func = `SUB;
            end
            else if (funct3 == `R_SLL)     ALU_func = `SLL;
            else if (funct3 == `R_SLT)     ALU_func = `SLT;
            else if (funct3 == `R_SLTU)    ALU_func = `SLTU;
            else if (funct3 == `R_XOR)     ALU_func = `XOR;
            else if (funct3 == `R_SR) begin
                if (funct7 == `R_SRL)      ALU_func = `SRL;
                else if (funct7 == `R_SRA) ALU_func = `SRA;
            end
            else if (funct3 == `R_OR)      ALU_func = `OR;
            else if (funct3 == `R_AND)     ALU_func = `AND;
            br_type = 0;
            load_npc = 0;
            wb_select = 0;
            load_type = 0;
            reg_write_en = 1;
            cache_write_en = 0;
            imm_type = `RTYPE;
            CSR_write_en = 0;
            CSR_zimm_or_reg = 0;
        end
        else if (opcode == `I_CSR) begin
            test = 32'd10;
            // `define I_CSRRC 3'b011
            // `define I_CSRRCI 3'b111
            // `define I_CSRRS 3'b010
            // `define I_CSRRSI 3'b110
            // `define I_CSRRW 3'b001
            // `define I_CSRRWI 3'b101
            if (funct3 == 3'b011 || funct3 == 3'b111) ALU_func = `CSRRC;
            else if (funct3 == 3'b010 || funct3 == 3'b110) ALU_func = `OR;
            else if (funct3 == 3'b001 || funct3 == 3'b101) ALU_func = `OP1;
            jal = 0;
            jalr = 0;
            op1_src = 0; // reg or imm
            op2_src = 0; // csr data
            // ALU_func = 0;
            br_type = 0;
            load_npc = 0;
            wb_select = 0; //选alu出即可
            load_type = 0;
            reg_write_en = 1;
            cache_write_en = 0;
            imm_type = 0;
            CSR_write_en = 1;
            // CSR_zimm_or_reg = 1; //将csr_zimm的值传给op1
            CSR_zimm_or_reg = funct3[2];
        end
        else begin
            test = 32'd11;
            jal = 0;
            jalr = 0;
            op1_src = 0;
            op2_src = 0;
            ALU_func = 0;
            br_type = 0;
            load_npc = 0;
            wb_select = 0;
            load_type = 0;
            reg_write_en = 0;
            cache_write_en = 0;
            imm_type = 0;
            CSR_write_en = 0;
            CSR_zimm_or_reg = 0; 
        end
    end

endmodule