from django.shortcuts import render

# Create your views here.
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from .forms import SignupForm
from django.contrib import messages

def signup_view(request):
    if request.method == "POST":
        form = SignupForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.set_password(form.cleaned_data["password1"])
            user.save()
            messages.success(request, "Đăng ký thành công!")
            return redirect("login")
    else:
        form = SignupForm()
    return render(request, "accounts/signup.html", {"form": form})


def login_view(request):
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")

        user = authenticate(request, username=username, password=password)
        if user:
            login(request, user)
            return redirect("home")  # TRỎ VỀ TRANG HOME CỦA BẠN
        else:
            messages.error(request, "Sai tài khoản hoặc mật khẩu!")

    return render(request, "accounts/login.html")


def logout_view(request):
    logout(request)
    return redirect("login")

from django.contrib.auth.decorators import login_required

@login_required
def home(request):
    return render(request, 'home.html')

