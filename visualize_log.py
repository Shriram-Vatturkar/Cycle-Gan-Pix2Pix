import re
from torch.utils.tensorboard import SummaryWriter

log_file_path =  r"C:\Users\scruk\pytorch-CycleGAN-and-pix2pix\loss_log.txt"  # replace with your path if needed
writer = SummaryWriter(log_dir="runs/loss_graph")

# Regular expression to match the log lines
pattern = re.compile(r"\(epoch: (\d+), iters: (\d+),.*?D_A: ([\d.]+) G_A: ([\d.]+) cycle_A: ([\d.]+) idt_A: ([\d.]+) D_B: ([\d.]+) G_B: ([\d.]+) cycle_B: ([\d.]+) idt_B: ([\d.]+)")

with open(log_file_path, "r") as f:
    for line in f:
        match = pattern.search(line)
        if match:
            epoch = int(match.group(1))
            iters = int(match.group(2))
            step = epoch * 10000 + iters  # custom global step

            writer.add_scalar("Loss/D_A", float(match.group(3)), step)
            writer.add_scalar("Loss/G_A", float(match.group(4)), step)
            writer.add_scalar("Loss/cycle_A", float(match.group(5)), step)
            writer.add_scalar("Loss/idt_A", float(match.group(6)), step)
            writer.add_scalar("Loss/D_B", float(match.group(7)), step)
            writer.add_scalar("Loss/G_B", float(match.group(8)), step)
            writer.add_scalar("Loss/cycle_B", float(match.group(9)), step)
            writer.add_scalar("Loss/idt_B", float(match.group(10)), step)

writer.close()
print("TensorBoard logs written to 'runs/loss_graph'")
