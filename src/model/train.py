import os
import sys
import models
from tqdm import tqdm
import tensorflow as tf
sys.path.append("../utils")
import visualization_utils as vu
import training_utils as tu
import data_utils as du
import objectives

FLAGS = tf.app.flags.FLAGS


def train_model():

    # Setup session
    sess = tu.setup_training_session()

    # Setup async input queue of real images
    X_real16, X_real32, X_real64 = du.read_celebA()

    #######################
    # Instantiate generators
    #######################
    G16 = models.G16()
    G32 = models.G32()
    G64 = models.G64()

    ###########################
    # Instantiate discriminators
    ###########################
    D16 = models.D16()
    D32 = models.D32()
    D64 = models.D64()

    ###########################
    # Instantiate optimizers
    ###########################
    G_opt = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate, name='G_opt', beta1=0.5)
    D_opt = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate, name='D_opt', beta1=0.5)

    ###########################
    # Instantiate model outputs
    ###########################

    # noise_input = tf.random_normal((FLAGS.batch_size, FLAGS.noise_dim,), stddev=0.1)
    noise_input = tf.random_uniform((FLAGS.batch_size, FLAGS.noise_dim,), minval=-1, maxval=1)

    X_fake16 = G16(noise_input)

    D16_real = D16(X_real16, mode="D")
    X_feat16, D16_fake = D16(X_fake16, reuse=True, mode="G")

    X_fake32 = G32(X_fake16, X_feat16)

    D32_real = D32(X_real32, mode="D")
    X_feat32, D32_fake = D32(X_fake32, reuse=True, mode="G")

    X_fake64 = G64(X_fake32, X_feat32)

    D64_real = D64(X_real64)
    D64_fake = D64(X_fake64, reuse=True)

    # output images
    X_fake16_output = du.unnormalize_image(X_fake16)
    X_real16_output = du.unnormalize_image(X_real16)

    X_fake32_output = du.unnormalize_image(X_fake32)
    X_real32_output = du.unnormalize_image(X_real32)

    X_fake64_output = du.unnormalize_image(X_fake64)
    X_real64_output = du.unnormalize_image(X_real64)

    ###########################
    # Instantiate losses
    ###########################

    G16_loss = objectives.binary_cross_entropy_with_logits(D16_fake, tf.ones_like(D16_fake))
    G32_loss = objectives.binary_cross_entropy_with_logits(D32_fake, tf.ones_like(D32_fake))
    G64_loss = objectives.binary_cross_entropy_with_logits(D64_fake, tf.ones_like(D64_fake))
    G_loss = G16_loss + G32_loss + G64_loss

    # Fake losses
    D16_loss_fake = objectives.binary_cross_entropy_with_logits(D16_fake, tf.zeros_like(D16_fake))
    D32_loss_fake = objectives.binary_cross_entropy_with_logits(D32_fake, tf.zeros_like(D32_fake))
    D64_loss_fake = objectives.binary_cross_entropy_with_logits(D64_fake, tf.zeros_like(D64_fake))

    # Real losses
    D16_loss_real = objectives.binary_cross_entropy_with_logits(D16_real, tf.ones_like(D16_real))
    D32_loss_real = objectives.binary_cross_entropy_with_logits(D32_real, tf.ones_like(D32_real))
    D64_loss_real = objectives.binary_cross_entropy_with_logits(D64_real, tf.ones_like(D64_real))

    D_loss = D16_loss_real + D32_loss_real + D64_loss_real
    D_loss += D16_loss_fake + D32_loss_fake + D64_loss_fake

    ###########################
    # Compute gradient updates
    ###########################

    dict_G16_vars = G16.get_trainable_variables()
    G16_vars = [dict_G16_vars[k] for k in dict_G16_vars.keys()]
    dict_G32_vars = G32.get_trainable_variables()
    G32_vars = [dict_G32_vars[k] for k in dict_G32_vars.keys()]
    dict_G64_vars = G64.get_trainable_variables()
    G64_vars = [dict_G64_vars[k] for k in dict_G64_vars.keys()]
    G_vars = G16_vars + G32_vars + G64_vars

    dict_D16_vars = D16.get_trainable_variables()
    D16_vars = [dict_D16_vars[k] for k in dict_D16_vars.keys()]
    dict_D32_vars = D32.get_trainable_variables()
    D32_vars = [dict_D32_vars[k] for k in dict_D32_vars.keys()]
    dict_D64_vars = D64.get_trainable_variables()
    D64_vars = [dict_D64_vars[k] for k in dict_D64_vars.keys()]
    D_vars = D16_vars + D32_vars + D64_vars

    G_gradvar = G_opt.compute_gradients(G_loss, var_list=G_vars, colocate_gradients_with_ops=True)
    G_update = G_opt.apply_gradients(G_gradvar, name='G_loss_minimize')

    D_gradvar = D_opt.compute_gradients(D_loss, var_list=D_vars, colocate_gradients_with_ops=True)
    D_update = D_opt.apply_gradients(D_gradvar, name='D_loss_minimize')

    ##########################
    # Summary ops
    ##########################

    # Add summary for gradients
    tu.add_gradient_summary(G_gradvar)
    tu.add_gradient_summary(D_gradvar)

    # Add scalar symmaries
    tf.summary.scalar("G16 loss", G16_loss)
    tf.summary.scalar("G32 loss", G32_loss)
    tf.summary.scalar("G64 loss", G64_loss)

    # Real losses
    tf.summary.scalar("D16 loss real", D16_loss_real)
    tf.summary.scalar("D32 loss real", D32_loss_real)
    tf.summary.scalar("D64 loss real", D64_loss_real)

    # Fake losses
    tf.summary.scalar("D16 loss fake", D16_loss_fake)
    tf.summary.scalar("D32 loss fake", D32_loss_fake)
    tf.summary.scalar("D64 loss fake", D64_loss_fake)

    summary_op = tf.summary.merge_all()

    ############################
    # Start training
    ############################

    # Initialize session
    saver = tu.initialize_session(sess)

    # Start queues
    du.manage_queues(sess)

    # Summaries
    writer = tu.manage_summaries(sess)

    # Run checks on data dimensions
    list_data = [noise_input]
    list_data += [X_fake16, X_fake32, X_fake64]
    list_data += [X_fake16_output, X_fake32_output, X_fake64_output]
    output = sess.run(list_data)
    tu.check_data(output, list_data)

    for e in tqdm(range(FLAGS.nb_epoch), desc="Training progress"):

        t = tqdm(range(FLAGS.nb_batch_per_epoch), desc="Epoch %i" % e, mininterval=0.5)
        for batch_counter in t:

                # Update D
            output = sess.run([D_update])

            # Update G
            output = sess.run([G_update])

            if batch_counter % (FLAGS.nb_batch_per_epoch // 20) == 0:
                output = sess.run([summary_op])
                writer.add_summary(output[-1], e * FLAGS.nb_batch_per_epoch + batch_counter)

            t.set_description('Epoch %i:' % e)

        # Plot some generated images
        output = sess.run([X_fake16_output, X_real16_output,
                           X_fake32_output, X_real32_output,
                           X_fake64_output, X_real64_output,])
        vu.save_image(output[:2], e=e, title="size_16")
        vu.save_image(output[2:4], e=e, title="size_32")
        vu.save_image(output[4:6], e=e, title="size_64")

        # Save session
        saver.save(sess, os.path.join(FLAGS.model_dir, "model"), global_step=e)

        # Show data statistics
        output = sess.run(list_data)
        tu.check_data(output, list_data)

    print('Finished training!')
