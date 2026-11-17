> Historical Phase 1 scope. For current setup and Phase 2/3 inference, see the phase-specific protocols and reproducibility guide.

# Mathematical foundations
For binary score $s$, $p=1/(1+e^{-s})$. For positive $t$, $s'=ts$ preserves the 0.5-threshold decision while changing $p$ unless $s=0$ or $t=1$. For multiclass softmax, $\mathrm{softmax}(z+c\mathbf{1})=\mathrm{softmax}(z)$; therefore absolute logits are not identifiable without a reference. Proper scoring rules detect some confidence changes hidden from accuracy. Statistical testing does not establish practical importance by itself.
