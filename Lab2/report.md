# Lab2实验报告

## PB21000164 来泽远

## CPU补全

### ALU

```verilog
always @ (*)
    begin
        case(ALU_func)
            `SLL: ALU_out = op1 << op2[4:0];
            `SRL: ALU_out = op1 >> op2[4:0];
            `SRA: ALU_out = $signed(op1) >>> op2[4:0];
            `ADD: ALU_out = op1 + op2;
            `SUB: ALU_out = op1 - op2;
            `XOR: ALU_out = op1 ^ op2;
            `OR: ALU_out = op1 | op2;
            `AND: ALU_out = op1 & op2;
            `SLT: ALU_out = ($signed(op1) < $signed(op2)) ? 32'd1 : 32'd0;
            `SLTU: ALU_out = (op1 < op2) ? 32'd1 : 32'd0;
            `LUI: ALU_out = op2;
            `OP1: ALU_out = op1;
            `OP2: ALU_out = op2;
            `NAND: ALU_out = ~(op1 & op2);
            `CSRRC: ALU_out = ~(op1) & op2;
            default: ALU_out = 32'b0;
        endcase
    end
```

1. 注意unsigned和signed的指令区别，verilog中默认为无符号
2. 添加了关于CSRRC的alu功能，即掩码位置0

### BranchDecision

做若干种比较即可

### ControllerDecoder

```verilog
if (opcode == ...) begin
	...
    if (funct3 == ...) begin
        ...
        if (funct7 == ...) begin
            ...
        end
        else if ...
    end
    else if ...
end
else if ...
```

1. 大体逻辑如上，按照指令手册一一判断即可
2. 需要注意，$JAL$的立即数使用$J$类编码而$JALR$的立即数使用$I$类编码

### DataExtend

```verilog
always @ (*)
    begin
        if (load_type == `NOREGWRITE || load_type == `LW) begin
            dealt_data = data;
        end
        else if (load_type == `LHU) begin
            case(addr)
                2'b00: dealt_data = {16'b0, data[15:0]};
                2'b10: dealt_data = {16'b0, data[31:16]};
                default : dealt_data = 32'b0;
            endcase
        end
        else if (load_type == `LH) begin
            case(addr)
                2'b00: dealt_data = {{16{data[15]}}, data[15:0]};
                2'b10: dealt_data = {{16{data[31]}}, data[31:16]};
                default : dealt_data = 32'b0;
            endcase
        end
        else if (load_type == `LBU) begin
            case(addr)
                2'b00: dealt_data = {24'b0, data[7:0]};
                2'b01: dealt_data = {24'b0, data[15:8]};
                2'b10: dealt_data = {24'b0, data[23:16]};
                2'b11: dealt_data = {24'b0, data[31:24]};
                default : dealt_data = 32'b0;
            endcase
        end
        else if (load_type == `LB) begin
            case(addr)
                2'b00: dealt_data = {{24{data[7]}}, data[7:0]};
                2'b01: dealt_data = {{24{data[15]}}, data[15:8]};
                2'b10: dealt_data = {{24{data[23]}}, data[23:16]};
                2'b11: dealt_data = {{24{data[31]}}, data[31:24]};
                default : dealt_data = 32'b0;
            endcase
        end
        else begin
            dealt_data = 32'b0;
        end
    end
```

1. 按长度分为word、half、byte，均放置于返回值的低位
2. 注意零扩展和符号扩展

### Hazard

前递逻辑略，对称补全即可。

```verilog
always @ (*) begin
        flushF  = 0;
        bubbleF = 0;
        flushD  = 0;
        bubbleD = 0;
        flushE  = 0;
        bubbleE = 0;
        flushM  = 0;
        bubbleM = 0;
        flushW  = 0;
        bubbleW = 0;
        if (rst) begin
            flushF  = 1;
            flushD  = 1;
            flushE  = 1;
            flushM  = 1;
            flushW  = 1;
        end
        else if (br || jalr) begin
            flushD = 1;
            flushE = 1;
        end
        else if (jal) begin
            flushD = 1;
        end
        else if (((reg1_srcD == reg_dstE  && reg1_srcD != 0) || (reg2_srcD == reg_dstE  && reg2_srcD != 0))  && wb_select ) begin
            flushE = 1;
            bubbleF = 1;
            bubbleD = 1;
        end
    end

```

1. 先处理br和jalr，在EX段，优先级最高，刷EX、ID。
2. 再处理jal，在ID段，刷ID。
3. 最后处理load-use，ID：use，EX：load，刷EX，阻塞IF、ID。在ID-EX段检测理论上与前两种情况不可能一起发生，不需要优先级，但是为了代码方便写在同一个always块中。

### ImmEXtend

按照指令手册完成即可，注意默认为位扩展

### NPCGenerator

```verilog
always @(*) begin
        if (jalr) 
            NPC = jalr_target;
        else if (br)
            NPC = br_target;
        else if (jal)
            NPC = jal_target;
        else
            NPC = PC; //已经加4了
    end
```

jalr和br同一个优先级，但是不会冲突，任意排布即可，jal次之，最后是默认PC+4。模块中传入的已经+4。

### CSR_EX

段间寄存器，按照其余寄存器一同实现即可。

### CSR_Regfile

```verilog
	reg [31:0] csr [127:0];
    integer i;
    initial
    begin
        for(i = 0; i < 4096; i = i + 1) 
            csr[i][31:0] <= 32'b0;
    end

    always@(posedge clk or posedge rst) 
    begin 
        if (rst)
            for (i = 0; i < 4096; i = i + 1) 
                csr[i][31:0] <= 32'b0;
        else if(CSR_write_en)
            csr[CSR_write_addr] <= CSR_data_write;   
    end
    assign CSR_data_read = csr[CSR_read_addr];
```

实现寄存器堆，注意指令手册规定CSR有极大的寻址空间。实际代码中CSR的容量过大会导致仿真极慢，够用即可。

## 问题记录

1. Decoder中部分指令译码不正确，主要是因为遗漏了对于funct7的判断

## 实验收获

对于流水线的理解更深，将课堂中讲述的难点如flush、stall，跳转等细节实现为代码。同时翻阅了基础的rsic-v指令集，对于指令集的架构也有了大体的认识。

实验用时约一个下午加一个晚上（debug），报告编写约半个小时。

## 改进意见

1. 所有段间寄存器均为上升沿触发，`GeneralRegsiter.v`中实现的通用寄存器为下降沿触发，在仿真中作为一个小trick起到了抢一个周期的作用。但我认为同步流水线中这样的写法是不规范的。此外，如果有同学或者后续有上板实现的需求，两种触发方式的寄存器可能导致上板行为与仿真不符。
2. 关于段间寄存器，我认为应该将初始的所有值都传递下去。关于PC可以设置两个寄存器，一个代表PC，一个代表跳转需要的PC-4，极大提高了代码可读性以及debug的方便程度。同时也可以在wb的同时设置一个提交段，既可以让同学们熟悉提交技术，看到哪些指令是真正执行了；也可以改进实验的检测方式为对比指令踪迹，比原本查寄存器的方式有更强的检错功能，同时也可以精确定位到哪一条指令出错。