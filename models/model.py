import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Dense, Concatenate

from models.layers import GraphConvolution, PatientFeatureEmbedding, SoftAttention, CategoryEmbedding
from models.layers import HierarchicalEmbedding, PatientEmbedding, WordEmbedding
from models.layers import VisitEmbedding, TemporalEmbedding, NoteEmbedding


class CCDLDFeatureExtractor(Layer):
    def __init__(self, config, hyper_params, name='ccdld_feature'):
        super().__init__(name=name)
        self.config = config
        self.num = 0
        self.hyper_params = hyper_params
        self.hierarchical_embedding_layer = HierarchicalEmbedding(
            code_levels=config['code_levels'],
            code_num_in_levels=config['code_num_in_levels'],
            code_dims=hyper_params['code_dims'])
        self.category_embedding_layer = CategoryEmbedding(
            code_category=config['code_category'],
            category_num=config['category_num'],
            category_dim=hyper_params['code_category_dim'])
        self.patient_embedding_layer = PatientEmbedding(
            patient_num=config['patient_num'],
            patient_dim=hyper_params['patient_dim'])
        self.patient_feature_embedding_layer = PatientFeatureEmbedding(
            patient_gender_dim=hyper_params['patient_gender_dim'],
            patient_age_dim=hyper_params['patient_age_dim'],
            patient_cluster_dim=hyper_params['patient_cluster_dim'])
        self.graph_convolution_layer = GraphConvolution(
            patient_dim=hyper_params['patient_dim'],
            code_dim=np.sum(hyper_params['code_dims']),
            patient_code_adj=config['patient_code_adj'],
            code_code_adj=config['code_code_adj'],
            patient_hidden_dims=hyper_params['patient_hidden_dims'],
            code_hidden_dims=hyper_params['code_hidden_dims'])
        self.visit_embedding_layer = VisitEmbedding(
            max_seq_len=config['max_visit_seq_len'])
        self.visit_temporal_embedding_layer = TemporalEmbedding(
            rnn_dims=hyper_params['visit_rnn_dims'],
            attention_dim=hyper_params['visit_attention_dim'],
            max_seq_len=config['max_visit_seq_len'],
            name='visit_temporal')
        if config['use_attention']:
            if config['use_note']:
                self.soft_attention = SoftAttention(hidden_size=hyper_params['soft_attention_dim'])
            else:
                self.soft_attention = SoftAttention(hidden_size=hyper_params['soft_attention_dim']/2)
        if config['use_note']:
            self.word_embedding_layer = WordEmbedding(
                word_num=config['word_num'],
                word_dim=hyper_params['word_dim'])
            self.note_embedding_layer = NoteEmbedding(
                attention_dim=hyper_params['note_attention_dim'],
                max_seq_len=config['max_note_seq_len'],
                lambda_=config['lambda'],
                name='note_embedding')

    def call(self, inputs, training=True):
        patient_gender = tf.reshape(inputs['patient_gender'], (-1, ))
        patient_age = tf.reshape(inputs['patient_age'], (-1, ))
        patient_cluster = tf.reshape(inputs['patient_cluster'], (-1, ))
        visit_codes = inputs['visit_codes']
        visit_lens = tf.reshape(inputs['visit_lens'], (-1, ))
        word_ids = inputs['word_ids']
        word_tf_idf = inputs['tf_idf_weight'] if training and self.config['use_note'] else None
        word_lens = tf.reshape(inputs['word_lens'], (-1, ))
        code_embeddings = self.hierarchical_embedding_layer(None)
        code_category_embeddings = self.category_embedding_layer(None)
        code_embeddings = code_embeddings * code_category_embeddings
        code_embeddings = tf.concat([code_embeddings, code_category_embeddings], axis=-1)
        patient_embeddings = self.patient_embedding_layer(None)
        patient_embeddings, code_embeddings = self.graph_convolution_layer(
            patient_embeddings=patient_embeddings, code_embeddings=code_embeddings)
        visits_embeddings = self.visit_embedding_layer(
            code_embeddings=code_embeddings,
            visit_codes=visit_codes,
            visit_lens=visit_lens)
        visit_output, alpha_visit = self.visit_temporal_embedding_layer(visits_embeddings, visit_lens)
        output = visit_output
        if self.config['use_note']:
            words_embeddings = self.word_embedding_layer(word_ids)
            note_output, alpha_word = self.note_embedding_layer(words_embeddings, word_lens, visit_output, word_tf_idf, training)
            note_output = tf.math.l2_normalize(note_output, axis=-1)
            visit_output = tf.math.l2_normalize(visit_output, axis=-1)
            output = Concatenate()([visit_output, note_output])
        stack_output = tf.stack(output)
        if self.config['use_attention']:
            output = self.soft_attention(stack_output)
        else:
            output = stack_output
        if self.config['use_patient_feature']:
            patient_feature_embeddings = self.patient_feature_embedding_layer(
                patient_gender=patient_gender,
                patient_age=patient_age,
                patient_cluster=patient_cluster)
            output = Concatenate()([output, patient_feature_embeddings])
        return output


class Classifier(Layer):
    def __init__(self, output_dim, activation=None, dropout=0.2, name='classifier'):
        super().__init__(name=name)
        self.dense = Dense(output_dim, activation=activation)
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x):
        x = self.dropout(x)
        output = self.dense(x)
        return output


class CCDLD(Model):
    def __init__(self, config, hyper_params, name='ccdld'):
        super().__init__(name=name)
        self.ccdld_feature_extractor = CCDLDFeatureExtractor(config, hyper_params)
        self.classifier = Classifier(config['output_dim'], activation=config['activation'], dropout=config['dropout'])

    def call(self, inputs, training=True):
        output = self.ccdld_feature_extractor(inputs, training=training)
        output = self.classifier(output)
        return output

