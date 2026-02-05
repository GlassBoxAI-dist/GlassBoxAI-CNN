/*
MIT License
Copyright (c) 2025 Matthew Abbott
*/

package cnn

import (
	"testing"
)

func TestVersion(t *testing.T) {
	v := Version()
	if v == "" {
		t.Error("Version() returned empty string")
	}
	t.Logf("Library version: %s", v)
}

func TestActivationTypeString(t *testing.T) {
	tests := []struct {
		act  ActivationType
		want string
	}{
		{Sigmoid, "sigmoid"},
		{Tanh, "tanh"},
		{ReLU, "relu"},
		{Linear, "linear"},
	}

	for _, tt := range tests {
		if got := tt.act.String(); got != tt.want {
			t.Errorf("ActivationType.String() = %v, want %v", got, tt.want)
		}
	}
}

func TestLossTypeString(t *testing.T) {
	tests := []struct {
		loss LossType
		want string
	}{
		{MSE, "mse"},
		{CrossEntropy, "crossentropy"},
	}

	for _, tt := range tests {
		if got := tt.loss.String(); got != tt.want {
			t.Errorf("LossType.String() = %v, want %v", got, tt.want)
		}
	}
}

// Note: CNN tests require CUDA hardware and the library to be built.
// Run with: go test -v -tags=integration
