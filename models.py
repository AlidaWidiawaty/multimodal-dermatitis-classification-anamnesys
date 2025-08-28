from torchvision.models import resnet50, ResNet50_Weights
from transformers import MPNetModel
import transformers
import torch.nn as nn
import torch
import timm


class MultimodalModel(nn.Module):
    def __init__(
        self, text_model_name="microsoft/mpnet-base", num_classes=2, hidden_size=768
    ):
        super(MultimodalModel, self).__init__()

        # Text model (AnamnesysModel)
        self.text_model = MPNetModel.from_pretrained(text_model_name)

        # Image model (SkinImageModel)
        # self.image_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        # self.image_model.fc = nn.Linear(self.image_model.fc.in_features, hidden_size)
        self.image_model = timm.create_model(
            "resnet50", pretrained=True, num_classes=768
        )

        # Final classifier (to process combined features)
        self.final_classifier = nn.Sequential(
            nn.Linear(
                2 * hidden_size, hidden_size
            ),  # Concatenated features (text + image)
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, input_ids, attention_mask=None, image=None):
        # Process the text data
        if attention_mask is not None:
            text_output = self.text_model(
                input_ids=input_ids, attention_mask=attention_mask
            )
        else:
            text_output = self.text_model(input_ids=input_ids)

        text_features = text_output.last_hidden_state[
            :, 0, :
        ]  # [CLS] token for classification

        # Process the image data
        image_features = self.image_model(image)

        # Concatenate text and image features
        combined_features = torch.cat((text_features, image_features), dim=1)

        # Final classification layer
        multimodal_logits = self.final_classifier(combined_features)

        return multimodal_logits


class MultimodalSkin(nn.Module):
    def __init__(
        self,
        text_model_name="microsoft/mpnet-base",
        num_classes=2,
        hidden_size=768,
        pretrained=False,
    ):
        super(MultimodalSkin, self).__init__()

        # Text model (AnamnesysModel)
        self.text_model = AnamnesysModel(text_model_name)

        # Image model (SkinImageModel)
        self.image_model = SkinImageModel()

        # Final classifier (to process combined features)
        self.final_classifier = nn.Sequential(
            nn.Linear(
                2 * hidden_size, hidden_size
            ),  # Concatenated features (text + image)
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, num_classes),
        )

        self.pretrained = pretrained

    def forward(self, input_ids, attention_mask=None, image=None):
        # Process the text data
        if attention_mask is not None:
            text_output = self.text_model(
                input_ids=input_ids, attention_mask=attention_mask
            )
        else:
            text_output = self.text_model(input_ids=input_ids)

        text_features = text_output.last_hidden_state[
            :, 0, :
        ]  # [CLS] token for classification

        # Process the image data
        image_features = self.image_model(image)

        # Concatenate text and image features
        combined_features = torch.cat((text_features, image_features), dim=1)

        # Final classification layer
        multimodal_logits = self.final_classifier(combined_features)

        return multimodal_logits


class AnamnesysModel(nn.Module):
    def __init__(
        self, model_name="microsoft/mpnet-base", num_classes=2, with_attentions=False
    ):
        super(AnamnesysModel, self).__init__()
        self.with_attentions = with_attentions
        # if self.with_attentions:
        #     self.mpnet_model = MPNetModel.from_pretrained(
        #         pretrained_model_name_or_path="microsoft/mpnet-base",
        #         output_hidden_states=True,
        #         output_attentions=True,
        #     )
        # else:
        #     self.mpnet_model = MPNetModel.from_pretrained(
        #         pretrained_model_name_or_path="microsoft/mpnet-base",
        #     )
        # print("CONFIG HIDDEN:", self.mpnet_model.config)
        self.mpnet_model = (
            transformers.AutoModelForSequenceClassification.from_pretrained(
                pretrained_model_name_or_path="microsoft/mpnet-base",
                output_hidden_states=True,
                output_attentions=True,
            )
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.mpnet_model.config.hidden_size, 512),  # Hidden layer
            nn.ReLU(),
            nn.Dropout(0.3),  # Dropout layer to prevent overfitting
            nn.Linear(
                512, num_classes
            ),  # Output layer (num_classes is the number of classes for classification)
        )

    def forward(self, input_ids, attention_mask=None):
        if attention_mask is not None:
            output = self.mpnet_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
            )
            # attentions = output.attentions
        else:
            output = self.mpnet_model(input_ids=input_ids, output_attentions=True)
            # attentions = output.attentions

        # print("OUTPUT SIZE: ", output.hidden_states[-1].shape)

        # Get the hidden state for the [CLS] token (first token) for classification
        # The output of the MPNet model contains several things, but we are interested in `last_hidden_state`
        # The shape of `last_hidden_state` is (batch_size, sequence_length, hidden_size)
        # print(output.last_hidden_state.shape)
        # hidden_state = output.last_hidden_state[
        #     :, 0, :
        # ]  # Extract the [CLS] token embedding
        # print("OUTPUT SIZE: ", output.hidden_states[-1].shape)

        logits = self.classifier(output.hidden_states[-1][:, 0, :])
        # logits = self.classifier(hidden_state)
        # print("LOGITS SIZE: ", logits.shape)

        # if self.with_attentions:
        #     return logits, attentions

        return logits


class SkinImageModel(nn.Module):
    def __init__(self, num_classes=2, hidden_size=512):
        super(SkinImageModel, self).__init__()
        # self.model = timm.create_model(
        #     "efficientnet_b0", pretrained=True, num_classes=2
        # )
        self.model = timm.create_model("resnet50", pretrained=True, num_classes=2)
        # self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        # self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x