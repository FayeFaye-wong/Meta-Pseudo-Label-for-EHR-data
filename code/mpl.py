import numpy as np
import pandas as pd
from trainer import CustomModel, ModelTrainer
from evaluation import Evaluation, eval_model
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm.notebook import tqdm
from accelerate import Accelerator


# Generate Pseudo Labels and combine with the labelled data
def pseudo(model,unlabelled_dl,labelled_dl):
    pseudo_labels = get_predictions(model, unlabelled_dl)
    return pseudo_labels

def save_checkpoint(config, state, is_teacher, is_best):
    if is_teacher:
        os.makedirs(config["teacher_model_dir"], exist_ok=True)
        name = config["teacher_save_name"]
        filename = f'{config["teacher_model_dir"]}/{name}_last.pth.tar'
    else:
        os.makedirs(config["student_model_dir"], exist_ok=True)
        name = config["student_save_name"]
        filename = f'{config["student_model_dir"]}/{name}_last.pth.tar'
    torch.save(state, filename, _use_new_zipfile_serialization=False)
    if is_best:
        if is_teacher:
            shutil.copyfile(filename, f'{config["teacher_model_dir"]}/{name}_best.pth.tar')
        else:
            shutil.copyfile(filename, f'{config["student_model_dir"]}/{name}_best.pth.tar')

class finetune:
    
    #------- load student and teacher into memory --------------------#
    student = CustomModel(hyperparameters).to(DEVICE)
    teacher = CustomModel(hyperparameters).to(DEVICE)

    #------- get optimizers for student and teacher ------------------#
    s_optimizer = optim.AdamW(student.parameters(), lr=hyperparameters['lr'])
    t_optimizer = optim.AdamW(teacher.parameters(), lr=hyperparameters['lr'])

   
    #The best scores for both the teacher and student models will be tracked, 
    #and a validation dataset is loaded from a specified path.
    best_teacher_score = 0
    best_student_score = 0

    valid_ds_path = os.path.join(config["output_dir"], config["valid_dataset_path"])
    valid_ds = load_from_disk(valid_ds_path)


    #------------- Data Iterators -------------------------------------#
    train_iter = iter(train_dl)
    unlabelled_iter = iter(unlabelled_dl)

    # ------- Training Loop  ------------------------------------------#
    for step in range(num_steps):

        #------ Reset buffers After Validation ------------------------#
        if step % config["validation_interval"] == 0:
            progress_bar = tqdm(range(min(config["validation_interval"], num_steps)))
            s_loss_meter = AverageMeter()
            t_loss_meter = AverageMeter()

        teacher.train()
        student.train()

        t_optimizer.zero_grad()
        s_optimizer.zero_grad()

        #------ Get Train & Unlabelled Batch -------------------------#
        try:
            train_b = train_iter.next()
        except Exception as e:  # TODO: change to stop iteration error
            train_b = next(train_dl.__iter__())

        try:
            unlabelled_b = unlabelled_iter.next()
        except:
            unlabelled_b = next(unlabelled_dl.__iter__())

        #------- Meta Training Steps ---------------------------------#
        # get loss of current student on labelled train data
        s_logits_train_b = student.get_logits(train_b)
        print(s_logits_train_b)
        # get loss of current student on labelled train data
        train_b_labels = train_b["labels"]
        train_b_masks = train_b_labels.gt(-0.5)
        s_loss_train_b = student.compute_loss(
            logits=s_logits_train_b.detach(),
            labels=train_b_labels,
            masks=train_b_masks
        )

        print("train")
        print(s_logits_train_b)
        print(train_b_labels)

        # get teacher generated pseudo labels for unlabelled data
        unlabelled_b_masks = unlabelled_b["label_mask"].eq(1).unsqueeze(-1)
        t_logits_unlabelled_b = teacher.get_logits(unlabelled_b)
        pseudo_y_unlabelled_b = (t_logits_unlabelled_b.detach() > 0).float()  # hard pseudo label

        #------ Train Student: With Pesudo Label Data ------------------#
        s_logits_unlabelled_b = student.get_logits(unlabelled_b)
        s_loss_unlabelled_b = student.compute_loss(
            logits=s_logits_unlabelled_b,
            labels=pseudo_y_unlabelled_b,
            masks=unlabelled_b_masks
        )   

        # backpropagation of student loss on unlabelled data
        accelerator.backward(s_loss_unlabelled_b)
        s_optimizer.step()  # update student params

        #------ Train Teacher ------------------------------------------#
        s_logits_train_b_new = student.get_logits(train_b)
        s_loss_train_b_new = student.compute_loss(
            logits=s_logits_train_b_new.detach(),
            labels=train_b_labels,
            masks=train_b_masks
        )
        change = s_loss_train_b_new - s_loss_train_b  # performance improvement from student

        t_logits_train_b = teacher.get_logits(train_b)
        t_loss_train_b = teacher.compute_loss(
            logits=t_logits_train_b,
            labels=train_b_labels,
            masks=train_b_masks
        )

        t_loss_mpl = change * F.binary_cross_entropy_with_logits(
            t_logits_unlabelled_b, pseudo_y_unlabelled_b, reduction='none')  # mpl loss
        t_loss_mpl = torch.masked_select(t_loss_mpl, unlabelled_b_masks).mean()
        t_loss = t_loss_train_b + t_loss_mpl

        # backpropagation of teacher's loss
        accelerator.backward(t_loss)
        t_optimizer.step()

        #------ Progress Bar Updates ----------------------------------#
        s_loss_meter.update(s_loss_train_b_new.item())
        t_loss_meter.update(t_loss.item())

        progress_bar.set_description(
            f"STEP: {step+1:5}/{num_steps:5}. "
            f"LR: {get_lr(s_optimizer):.4f}. "
            f"TL: {t_loss_meter.avg:.4f}. "
            f"SL: {s_loss_meter.avg:.4f}. "
        )
        progress_bar.update()

        #------ Evaluation & Checkpointing -----------------------------#
        if (step + 1) % config["validation_interval"] == 0:
            progress_bar.close()

            #----- Teacher Evaluation  ---------------------------------#
            teacher.eval()
            teacher_preds = []
            with torch.no_grad():
                for batch in valid_dl:
                    p = teacher.get_logits(batch)
                    teacher_preds.append(p)
            teacher_preds = [torch.sigmoid(p).detach().cpu().numpy()[:, :, 0] for p in teacher_preds]
            teacher_preds = list(chain(*teacher_preds))
            teacher_lb = scorer_fn(teacher_preds)
            print(f"After step {step+1} Teache LB: {teacher_lb}")

            # save teacher
            accelerator.wait_for_everyone()
            teacher = accelerator.unwrap_model(teacher)
            teacher_state = {
                'step': step + 1,
                'state_dict': teacher.state_dict(),
                'optimizer': t_optimizer.state_dict(),
                'lb': teacher_lb
            }
            is_best = False
            if teacher_lb > best_teacher_score:
                best_teacher_score = teacher_lb
                is_best = True
            # save_checkpoint(config, teacher_state, is_teacher=True, is_best=is_best)

            #----- Student Evaluation  ---------------------------------#
            student.eval()
            student_preds = []
            with torch.no_grad():
                for batch in valid_dl:
                    p = student.get_logits(batch)
                    student_preds.append(p)
            student_preds = [torch.sigmoid(p).detach().cpu().numpy()[:, :, 0] for p in student_preds]
            student_preds = list(chain(*student_preds))
            student_lb = scorer_fn(student_preds)
            print(f"After step {step+1} Student LB: {student_lb}")

            # save student
            accelerator.wait_for_everyone()
            student = accelerator.unwrap_model(student)
            student_state = {
                'step': step + 1,
                'state_dict': student.state_dict(),
                'optimizer': s_optimizer.state_dict(),
                'lb': student_lb
            }
            is_best = False
            if student_lb > best_student_score:
                best_student_score = student_lb
                is_best = True
            save_checkpoint(config, student_state, is_teacher=False, is_best=is_best)

