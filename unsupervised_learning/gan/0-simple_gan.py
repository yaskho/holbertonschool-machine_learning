def train_step(self, useless_argument):
    # Train the discriminator
    for _ in range(self.disc_iter):
        with tf.GradientTape() as tape:
            real_sample = self.get_real_sample()
            fake_sample = self.get_fake_sample(training=True)

            real_pred = self.discriminator(real_sample, training=True)
            fake_pred = self.discriminator(fake_sample, training=True)

            discr_loss = self.discriminator.loss(real_pred, fake_pred)

        gradients = tape.gradient(
            discr_loss,
            self.discriminator.trainable_variables
        )

        self.discriminator.optimizer.apply_gradients(
            zip(gradients, self.discriminator.trainable_variables)
        )

    # Train the generator
    with tf.GradientTape() as tape:
        fake_sample = self.get_fake_sample(training=True)
        fake_pred = self.discriminator(fake_sample, training=True)

        gen_loss = self.generator.loss(fake_pred)

    gradients = tape.gradient(
        gen_loss,
        self.generator.trainable_variables
    )

    self.generator.optimizer.apply_gradients(
        zip(gradients, self.generator.trainable_variables)
    )

    return {
        "discr_loss": discr_loss,
        "gen_loss": gen_loss
    }
