use crate::error::Result;

pub fn run(path: String, name: String) -> Result<()> {
    println!("base at {path}");

    println!("base at {path}, {name}");
    base_method();

    Ok(())
}

macro_rules! say_hello {
    () => {
        println!("Hello, world!");
    };
    ($name:expr) => {
        println!("Hello, {}!", $name);
    };
}

fn base_method(){

    let x = 4;
    // x = 3;
    let mut y = 19;
    y = 33;

    println!("{}", y);
    let s = "hello world";

    if y > 10 {
        println!("y > 10");
    }else {
        println!("y < 10");
    }

    for i in 1..4{
        println!("{}", i)
    }

    let s1 = String::from("hello");

    // const s5: String = "ddd";
    let s2 = &s1;

    println!("{}, {}", s1, s2);

    // let s3 = s1;
    println!("{}", s1);

    say_hello!(1);
    say_hello2(3);
}

fn say_hello2(a: i32){

}

#[derive(Debug, Clone, PartialEq)]
struct Point {
    x: i32,
    y: i32,
}

// impl std::fmt::Debug for Point {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         write!(f, "Point {{ x: {}, y: {} }}", self.x, self.y)
//     }
// }