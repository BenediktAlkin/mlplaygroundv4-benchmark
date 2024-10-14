import torch
import torch.nn as nn
from kappaschedules import object_to_schedule
from utils.schedule_utils import get_value_or_default


class RandomPartition(nn.Module):
    def __init__(self, num_crops, num_partitions):
        super().__init__()
        self.num_crops = num_crops
        self.num_partitions = num_partitions

    def forward(self, student_output, teacher_output):
        student_out = student_output.chunk(self.num_crops)
        teacher_out = teacher_output.detach().chunk(2)
        assert student_out[0].shape == teacher_out[0].shape

        #number_of_partitions = self.num_partitions

        # logic for rangom partioning into subgroups
        # rand_cluster_indices = torch.multinomial(
        #     torch.ones(
        #         [
        #             student_out[0].shape[-1],
        #         ],
        #         dtype=torch.float,
        #     ),
        #     number_of_partitions * partition_size,
        #     replacement=False,
        # ).cuda()

        num_prototypes = teacher_out[0].shape[1]
        rand_cluster_indices = torch.randperm(num_prototypes, device=teacher_out[0].device)

        partition_size = num_prototypes // self.num_partitions
        split_cluster_ids = torch.stack(
            torch.split(rand_cluster_indices, partition_size)
        )

        probs_list = []
        for log_view in student_out:
            predictions_group = self.get_logits_group(
                log_view, split_cluster_ids, partition_size
            )
            probs_list.append(predictions_group)

        targets_list = []
        for tar_view in teacher_out:
            targets_group = self.get_logits_group(
                tar_view, split_cluster_ids, partition_size
            )
            targets_list.append(targets_group)

        return probs_list, targets_list

    def get_logits_group(self, logits, split_cluster_ids, partition_size):
        logits_group = logits[:, split_cluster_ids.flatten()]
        logits_group = logits_group.split(partition_size, dim=1)
        logits = torch.stack(logits_group, dim=0)  ## [N_BLOCKS * BS, BLOCK_SIZE]
        return logits


class MasslLoss(nn.Module):
    def __init__(
            self,
            teacher_temperature,
            num_partitions,
            student_temperature=0.1,
            teacher_temperature_schedule=None,
            num_crops=12,
            update_counter=None,
    ):
        super().__init__()
        self.num_partitions = num_partitions
        self.student_temperature = student_temperature
        self.teacher_temperature = teacher_temperature
        if teacher_temperature_schedule is not None:
            assert update_counter is not None
            self.teacher_temperature_schedule = object_to_schedule(
                teacher_temperature_schedule,
                batch_size=update_counter.effective_batch_size,
                updates_per_epoch=update_counter.updates_per_epoch,
                max_value=teacher_temperature,
            )
        else:
            self.teacher_temperature_schedule = None
        self.partitioning = RandomPartition(
            num_crops=num_crops,
            num_partitions=num_partitions
        )
        self.update_counter = update_counter

    def cross_entropy(self, p, q):
        # assert inputs.shape == targets.shape
        #assert p.requires_grad
        assert not q.requires_grad

        p = torch.log_softmax(p, dim=-1)
        q = torch.softmax(q, dim=-1)

        loss = torch.sum(-q * p, dim=-1).mean()
        return loss

    def forward(self, student_output, teacher_output, reduction):
        assert reduction == "mean"

        student_output = student_output / self.student_temperature
        # TODO last eval step always crashes
        try:
            teacher_temperature = get_value_or_default(
                default=self.teacher_temperature,
                schedule=self.teacher_temperature_schedule,
                update_counter=self.update_counter,
                training=self.training,
            )
        except:
            print("ERROR in get_value_or_default")
            teacher_temperature = 0.07
        teacher_output = teacher_output / teacher_temperature

        if student_output.shape != teacher_output.shape:
            student_output, teacher_output = self.partitioning(
                student_output=student_output,
                teacher_output=teacher_output,
            )
        else:
            # hack for eval mode (mode is not propagated to loss)
            student_output = [student_output, student_output]
            teacher_output = [teacher_output, teacher_output]

        consistency = 0
        count = 0
        for i in range(len(student_output)):
            for j in range(len(teacher_output)):
                if i == j:
                    continue
                consistency += self.cross_entropy(student_output[i], teacher_output[j])
                count += 1

        consistency /= count
        return dict(total=consistency), {}
