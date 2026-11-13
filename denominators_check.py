M = {
 "Train A -> Test A (held-out tiles)": [[0,0],[0,0]],
 "Train A -> Test B (whole region)":   [[0,0],[0,0]],
 "Train B -> Test B (held-out tiles)": [[0,0],[0,0]],
 "Train B -> Test A (whole region)":   [[0,0],[0,0]],
}
print("evaluation | n | built-up % | acc | recall | spec | bal.acc | macroF1 | IoU")
for k,((tn,fp),(fn,tp)) in M.items():
    n = tn+fp+fn+tp
    if n == 0: continue
    rec = tp/(tp+fn); spec = tn/(tn+fp); prec = tp/(tp+fp)
    f1b = 2*prec*rec/(prec+rec)
    prec0 = tn/(tn+fn); f10 = 2*prec0*spec/(prec0+spec)
    print(f"{k} | {n:,} | {100*(tp+fn)/n:.1f} | {(tp+tn)/n:.3f} | {rec:.3f} | {spec:.3f} | {(rec+spec)/2:.3f} | {(f1b+f10)/2:.3f} | {tp/(tp+fp+fn):.3f}")
