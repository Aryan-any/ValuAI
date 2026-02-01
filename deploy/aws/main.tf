terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.16"
    }
  }
  required_version = ">= 1.2.0"
}

provider "aws" {
  region  = "us-east-1"
}

resource "aws_security_group" "valuai_sg" {
  name        = "valuai_security_group"
  description = "Allow HTTP and SSH traffic"

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"] 
    # In production, secure this to your IP only!
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_instance" "valuai_server" {
  ami           = "ami-0261755bbcb8c4a84" # Ubuntu 20.04 LTS (Update for your region!)
  instance_type = "t2.micro"              # Free Tier Eligible

  key_name      = var.key_name
  security_groups = [aws_security_group.valuai_sg.name]

  user_data = <<-EOF
              #!/bin/bash
              sudo apt-get update
              sudo apt-get install -y docker.io git
              sudo systemctl start docker
              sudo systemctl enable docker
              sudo usermod -aG docker ubuntu
              
              # Clone Repo (Public)
              git clone https://github.com/Aryan-any/ValuAI.git /home/ubuntu/valuai
              cd /home/ubuntu/valuai
              
              # Build & Run (Simplified)
              # Note: In real production, use ECR or Docker Hub image
              docker build -t valuai:latest .
              docker run -d -p 80:8000 \
                --env-file .env.example \
                valuai:latest
              EOF

  tags = {
    Name = "ValuAI-Production-Server"
  }
}

variable "key_name" {
    description = "Name of the SSH key pair"
    type        = string
}

output "public_ip" {
  value       = aws_instance.valuai_server.public_ip
  description = "The public IP address of the ValuAI server"
}
