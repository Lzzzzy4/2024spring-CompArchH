# Lab1 Report

## 3.24	PB21000164 来泽远

### 描述执行一条 XOR 指令的过程（数据通路、控制信号等）

IF：pc输入至cache中，进行取指，取出XOR指令，输入至IR寄存器。

ID：IR的op部分输入至Decoder，译码为XOR，输出控制信号传入Control寄存器，Op1和Op2均选择寄存器，RegWrite为1，ALU Func为XOR，WBselect选择alu；同时IR的Src部分传入寄存器堆，读出两个寄存器的值传入OP1、OP2寄存器。

EX：ALU执行ALUout = Op1 ^ Op2，传入RESULT寄存器。

MEM：WBMUX选择alu结果，存入WBDATA寄存器。

WB：按照写回地址写回。

### 描述执行一条 BEQ 指令的过程·（数据通路、控制信号等）

IF：pc输入至cache中，进行取指，取出BEQ指令，输入至IR寄存器。

ID：IR的op部分输入至Decoder，译码为BEQ，输出控制信号传入Control寄存器，Op1和Op2均选择寄存器，RegWrite为1，BrType为EQ；同时IR的Src部分传入寄存器堆，读出两个寄存器的值传入OP1、OP2寄存器；IMM单元生成立即数，左移后与pc相加得出跳转地址存入Br Target。

EX：Branch Module判断op1和op2是否相等，若相等则npc选择为br，否则为pc+4，在下一周期存入。

### 描述执行一条 LHU 指令的过程（数据通路、控制信号等）

IF：pc输入至cache中，进行取指，取出LHU指令，输入至IR寄存器。

ID：IR的op部分输入至Decoder，译码为XOR，输出控制信号传入Control寄存器，Op1选择寄存器，Op2选择立即数，RegWrite为1，ALU Func为ADD，WBselect选择mem，Load TypeM为HU；同时IR的Src部分传入寄存器堆，读出两个寄存器的值传入OP1、OP2寄存器。

EX：ALU执行ALUout = Op1 + Op2，传入RESULT寄存器。

MEM：cache根据RESULT寄存器中的地址取数（16位），经Extension零扩展后传入WBDATA。

WB：按照写回地址写回。

### 如果要实现 CSR 指令（csrrw，csrrs，csrrc，csrrwi，csrrsi，csrrci），设计图中还需要增加什么部件和数据通路？给出详细说明。

需要两个模块CSRreg，Priv。CSRreg放在ID段，与寄存器堆同时读写。Priv放在EX段，用于计算csr_wdata。

这五条指令本质就是把csr寄存器的值读入rd，再计算出一个值写入csr。大体需要增加的信号和通路如下：

1. 读写csr的信号（we、addr、wdata等）、在ex段的result前增加一个mux用于选择csr。
2. Priv的控制信号，输入包含读出的csr、读出的reg、一个立即数，大体与ALU的通路类似，增加一个口读入csr即可。
3. 关于上述信号的若干段间寄存器。

### Verilog 如何实现立即数的扩展？

使用拼接，零拓展：$imm_{extension}[31:0] = \{20'd0,imm[11:0]\}$

位扩展：$imm_{extension}[31:0] = \{20\{imm[11]\},imm[11:0]\}$

### 如何实现 Data Memory 的非字对齐的 Load 和 Store？

AXI总线中存取不强制地址非字对齐，所以有两种实现方式。

1. 约定按字对齐，则将地址后两位截取，存数和取数都从32位的data的对应8或者16位存或者取，存时还需配置wstrb。
2. 不约定按字对齐，直接从data的低8位或者16位存取即可。

以上两种方法都需要正确配置arlen、awlen等参数，使用sram协议也是类似两种方法。

### ALU 模块中，默认 wire 变量是有符号数还是无符号数？

无符号数。

### 简述 BR 信号的作用。

BR信号用于配合比较后跳转的指令，若满足跳转条件则BR置1，会在NPC MUX中选择BR Target，而后写入pc完成跳转。

### NPC Generator 中对于不同跳转 target 的选择有没有优先级？

有，jalr=br>jal。

jalr和br都在EX段执行，jal在ID段执行。设想一条jalr（或者br）后跟着一条jal，则MUX处多个信号有效，应该判流水级在后的信号有效。

### Harzard 模块中，有哪几类冲突需要插入气泡，分别使流水线停顿几个周期？

load and use：1个

### Harzard 模块中采用静态分支预测器，即默认不跳转，遇到 branch指令时，如何控制 flush 和 stall 信号？

branch在ex执行时，置IF-ID、ID-EX的所有段间寄存器flush为1

### 0 号寄存器值始终为 0，是否会对 forward 的处理产生影响？

会，如向0号寄存器写入1，该1在WB阶段，可能会被不正确地前递。所以在forward单元中需要特判src为0号寄存器时不前递，直接读0。