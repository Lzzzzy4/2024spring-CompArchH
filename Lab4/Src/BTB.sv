module BTB #(    
    parameter  BUF_LEN =  100
)(
    input clk, rst,
    input [31:0] pc,
    output reg [31:0] pc_predict,
    output reg valid_predict,
    input [31:0] pc_ex,
    input is_branch,
    input branch,
    input [31:0] branch_pc
    
);
reg [31:0]pc_tbl[BUF_LEN];
reg [31:0]branch_target[BUF_LEN];
reg [BUF_LEN - 1:0]state;
reg [BUF_LEN - 1:0]valid;
reg [31:0] pointer;

reg [BUF_LEN - 1:0]hit;
reg [BUF_LEN - 1:0]hit_exe;
always @(*) begin
    for (integer i = 0; i < BUF_LEN; i++) begin
        hit[i] = valid[i] && pc_tbl[i] == pc;
        hit_exe[i] = valid[i] && pc_tbl[i] == pc_ex;
    end
end

always @(*) begin
    pc_predict = 0;
    valid_predict = 0;
    for (integer i = 0; i < BUF_LEN; i++) begin
        if (hit[i] && state[i]) begin
            pc_predict = branch_target[i];
            valid_predict = 1;
        end
    end
end

always @ (posedge clk or posedge rst) begin
    if (rst) begin
        for(integer i = 0; i < BUF_LEN; i++) begin
            pc_tbl[i] <= 0;
            branch_target[i] <= 0;
            state[i] <= 0;
            valid[i] <= 0;
        end
        pointer <= 0;
    end
    else if (is_branch) begin
        for (integer i = 0; i < BUF_LEN; i++) begin
            if (hit_exe[i]) begin
                branch_target[i] <= branch_pc;
                state[i] <= branch;
            end
        end
        if (~|hit_exe) begin
            pc_tbl[pointer] <= pc_ex;
            branch_target[pointer] <= branch_pc;
            state[pointer] <= branch;
            valid[pointer] <= 1;
            pointer <= pointer == BUF_LEN - 1 ? 0 : pointer + 1;
        end
    end
end

endmodule