# Generated by Django 4.0.3 on 2022-04-19 22:25

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='BooleanSearch',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('word_dict', models.JSONField()),
                ('term_vectors', models.JSONField()),
            ],
        ),
    ]