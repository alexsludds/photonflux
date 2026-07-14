`ifndef DISCIPLINE_H
`define DISCIPLINE_H

// Minimal electrical discipline for ADMS
nature Voltage
    units = "V";
    access = V;
endnature

nature Current
    units = "A";
    access = I;
endnature

discipline electrical
    potential Voltage;
    flow Current;
enddiscipline

`endif
