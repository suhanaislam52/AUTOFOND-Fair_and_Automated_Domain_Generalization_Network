import torch
import torch.nn.functional as F

from src.networks.fond import FONDBase


class FOND_Distillation(FONDBase):
    def __init__(self, input_shape, num_classes, num_domains, hparams, teacher):
        super(FOND_Distillation, self).__init__(
            input_shape, num_classes, num_domains, hparams
        )
        self.teacher_setting = None
        self.distillation_temperature = hparams["distillation_temperature"]
        self.teacher = teacher

        # Freeze the parameters of the teacher network
        for param in self.teacher.classifier.parameters():
            param.requires_grad = False
        for param in self.teacher.featurizer.parameters():
            param.requires_grad = False
        for param in self.teacher.projector.parameters():
            param.requires_grad = False

    def preprocess(self, minibatches):
        # NOTE: current implementations doesn't create duplicates
        # check SelfReg

        num_domains = len(minibatches)
        features_student = [self.featurizer(xi) for xi, _ in minibatches]
        features_teacher = [self.teacher.featurizer(xi) for xi, _ in minibatches]

        if self.teacher_setting == "separate_projector":
            projections_student = [
                F.normalize(self.projector(fi)) for fi in features_student
            ]
            projections_teacher = [
                F.normalize(self.teacher.projector(fi)) for fi in features_teacher
            ]
        elif self.teacher_setting == "teacher_projector":
            projections_student = [
                F.normalize(self.teacher.projector(fi)) for fi in features_student
            ]
            projections_teacher = [
                F.normalize(self.teacher.projector(fi)) for fi in features_teacher
            ]
        elif self.teacher_setting == "student_projector":
            projections_student = [
                F.normalize(self.projector(fi)) for fi in features_student
            ]
            projections_teacher = [
                F.normalize(self.projector(fi)) for fi in features_teacher
            ]
        else:
            raise ValueError("Invalid teacher setting")

        classifs = [self.classifier(fi) for fi in features_student]
        targets = [yi for _, yi in minibatches]

        # create domain labels
        domains = [
            torch.zeros(len(x), dtype=torch.uint8).to(x.device) + i
            for i, x in enumerate(targets)
        ]

        # match domains
        if self.domain_relations is not None:
            for old, new in self.domain_relations.items():
                for d in domains:
                    d[d == old] = new

        return {
            "features_student": features_student,
            "features_teacher": features_teacher,
            "projections_student": projections_student,
            "projections_teacher": projections_teacher,
            "classifs": classifs,
            "targets": targets,
            "domains": domains,
            "num_domains": num_domains,
        }

    def update(self, minibatches, unlabeled=None):
        values = self.preprocess(minibatches)

        domains = torch.cat(values["domains"])
        targets = torch.cat(values["targets"])
        classifs = torch.cat(values["classifs"])

        features_student = torch.cat(values["features_student"])
        features_teacher = torch.cat(values["features_teacher"])
        projections_student = torch.cat(values["projections_student"])
        projections_teacher = torch.cat(values["projections_teacher"])

        noc_mask = self.noc_weight.to(targets.device)[targets].type(torch.bool)
        soft_projections_student = F.softmax(
            projections_student[noc_mask] / self.distillation_temperature
        )
        soft_projections_teacher = F.softmax(
            projections_teacher[noc_mask] / self.distillation_temperature
        )

        teacher_loss = F.kl_div(soft_projections_student, soft_projections_teacher)

        masks = self.get_masks(Y=targets, D=domains)

        alpha = (
            ~masks["diff_domain_same_class_mask"]
            + masks["diff_domain_same_class_mask"] * self.xda_alpha
        )

        beta = (
            ~masks["same_domain_diff_class_mask"]
            + masks["same_domain_diff_class_mask"] * self.xda_beta
        )

        xdom_loss, mean_positives_per_sample, num_zero_positives = self.supcon_loss(
            projections=projections_student,
            positive_mask=masks["same_class_mask"] * masks["self_mask"],
            negative_mask=masks["self_mask"],
            alpha=alpha,
            beta=beta,
        )

        class_loss = F.cross_entropy(classifs, targets)
        if torch.isnan(class_loss):
            class_loss = torch.tensor(0).to(targets.device)
        if torch.isnan(teacher_loss):
            teacher_loss = torch.tensor(0).to(targets.device)

        loss = class_loss + teacher_loss + self.xdom_lmbd * xdom_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "class_loss": class_loss.item(),
            "xdom_loss": xdom_loss.item(),
            "teacher_loss": teacher_loss.item(),
            "mean_p": mean_positives_per_sample.item(),
            "zero_p": num_zero_positives.item(),
        }


class FOND_Distillation_Separate_Projector(FOND_Distillation):
    def __init__(self, input_shape, num_classes, num_domains, hparams, teacher):
        super(FOND_Distillation_Separate_Projector, self).__init__(
            input_shape, num_classes, num_domains, hparams, teacher
        )
        self.teacher_setting = "separate_projector"


class FOND_Distillation_Teacher_Projector(FOND_Distillation):
    def __init__(self, input_shape, num_classes, num_domains, hparams, teacher):
        super(FOND_Distillation_Teacher_Projector, self).__init__(
            input_shape, num_classes, num_domains, hparams, teacher
        )
        self.teacher_setting = "teacher_projector"


class FOND_Distillation_Student_Projector(FOND_Distillation):
    def __init__(self, input_shape, num_classes, num_domains, hparams, teacher):
        super(FOND_Distillation_Student_Projector, self).__init__(
            input_shape, num_classes, num_domains, hparams, teacher
        )
        self.teacher_setting = "student_projector"
