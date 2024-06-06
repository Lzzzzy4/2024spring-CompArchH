`timescale 1ns / 1ps
//  功能说明
    //  根据跳转信号，决定执行的下一条指令地址
    //  debug端口用于simulation时批量写入数据，可以忽略
// 输入
    // PC                指令地址（PC + 4, 而非PC）
    // jal_target        jal跳转地址
    // jalr_target       jalr跳转地址
    // br_target         br跳转地址
    // jal               jal == 1时，有jal跳转
    // jalr              jalr == 1时，有jalr跳转
    // br                br == 1时，有br跳转
// 输出
    // NPC               下一条执行的指令地址
// 实验要求  
    // 实现NPC_Generator

module NPC_Generator(
    input clk, rst,
    input [31:0]pc_ex,pc_id,
    input is_branch,

    input bubbleF, bubbleD, bubbleE,
    input flushF, flushD, flushE,

    input wire [31:0] PC, jal_target, jalr_target, br_target,
    input wire jal, jalr, br,
    output reg br_for_harzard,
    output reg [31:0] NPC
    );

    // TODO: Complete this module

    /* FIXM: Write your code here... */
    wire [31:0] pc_predict;
    wire valid;

    // BTB BTB(
    BHT BHT(
        .clk(clk),
        .rst(rst),
        .pc(PC),
        .pc_ex(pc_ex),
        .is_branch(is_branch),
        .branch(br),
        .branch_pc(br_target),
        .pc_predict(pc_predict),
        .valid_predict(valid)
    );

    // static
    // assign valid = 0;
    // assign pc_predict = PC + 4;

    reg valid_F, valid_D, valid_E;
    reg [31:0]pc_predict_F, pc_predict_D, pc_predict_E;
    reg [31:0]pc_4_F, pc_4_D, pc_4_E;
    reg [5:0]test;

    reg [31:0] right;
    reg [31:0] all;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            right <= 0;
            all <= 0;
        end
        else begin
            if (!bubbleE)begin
                if (is_branch) begin
                    all <= all + 1;
                    if ((br && valid_D) || (!br && !valid_D)) begin
                        right <= right + 1;
                    end
                end
            end
        end
    end

    always @(*) begin
        br_for_harzard = 0;
        test = 0;
        if (jalr) begin
            NPC = jalr_target;
            test[0] = 1;
        end
        else if (is_branch && br && !valid_D) begin
                br_for_harzard = 1;
                NPC = br_target;
                test[1] = 1;
        end
        else if (is_branch && !br && valid_D) begin
                br_for_harzard = 1;
                NPC = pc_4_D;
                test[2] = 1;
        end
        else if (jal) begin
            NPC = jal_target;
            test[3] = 1;
        end
        else if (valid) begin
            NPC = pc_predict;
            test[4] = 1;
        end
        else begin
            NPC = PC + 32'd4;
            test[5] = 1;
        end
    end
    
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            valid_F <= 0;
            pc_predict_F <= 0;
            pc_4_F <= 0;
            valid_D <= 0;
            pc_predict_D <= 0;
            pc_4_D <= 0;
            valid_E <= 0;
            pc_predict_E <= 0;
            pc_4_E <= 0;
        end          
        else begin
            if (!bubbleF) begin
                if (flushF) begin
                    valid_F <= 0;
                    pc_predict_F <= 0;
                    pc_4_F <= 0;
                end
                else begin
                    valid_F <= valid;
                    pc_predict_F <= pc_predict;
                    pc_4_F <= PC + 4;
                end
            end
            if (!bubbleD) begin
                if (flushD) begin
                    valid_D <= 0;
                    pc_predict_D <= 0;
                    pc_4_D <= 0;
                end
                else begin
                    valid_D <= valid_F;
                    pc_predict_D <= pc_predict_F;
                    pc_4_D <= pc_4_F;
                end
            end
            if (!bubbleE) begin
                if (flushE) begin
                    valid_E <= 0;
                    pc_predict_E <= 0;
                end
                else begin
                    valid_E <= valid_D;
                    pc_predict_E <= pc_predict_D;
                    pc_4_E <= pc_4_D;
                end
            end
        end
    end


endmodule