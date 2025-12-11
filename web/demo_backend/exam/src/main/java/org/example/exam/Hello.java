package org.example.exam;

import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import javax.swing.*;

//@CrossOrigin(origins = "*")
@RestController
public class Hello {
    @RequestMapping
    String getHello() {
        return "hello world";
    }
}
