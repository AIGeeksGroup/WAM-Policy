# The LIBERO-Pro2.0 evaluation protocol of the paper; sourced by setup.sh and libero_pro2.sh.
# The official scored selection of the evaluation package, restricted to LIBERO-10 (held-out tasks 5 and 8),
# with LingBot-VA's LIBERO client settings. --max-steps 800 is that client's limit (the package allows the
# override when recorded); the official summary also reports the package default of 520 steps.
PLAN_ARGS=(--split test --suite libero_10 --group all --seed 142 --num-seeds 30 --init-state-ids 0
           --control-mode relative --camera-height 128 --camera-width 128 --max-steps 800)
# 84 task instances (2 BASE, 43 static, 39 runtime) x 30 seeds; 6 distinct instructions. PLAN_ID is the id of the
# paper's plan, printed by setup.sh for comparison.
PLAN_ID=29992fd5e8d5246fde0f1bd023cee159f01f74992d6720c8b4854274c6f325b1
PLAN_ROLLOUTS=2520
PLAN_TEXTS=6
LIBERO_COMMIT=8f1084e3132a39270c3a13ebe37270a43ece2a01
